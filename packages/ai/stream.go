package ai

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/holbrookab/go-ai/internal/retry"
)

var (
	smoothStreamWordRE = regexp.MustCompile(`\S+\s+`)
	smoothStreamLineRE = regexp.MustCompile(`\n+`)
)

func StreamText(ctx context.Context, opts StreamTextOptions) (result *StreamTextResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventStreamTextStart, OperationStreamText, opts.Model, nil)
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventStreamTextError, OperationStreamText, err)
		}
	}()
	if opts.Model == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "model is required"}
	}
	var cancel context.CancelFunc
	if opts.Timeout.Total > 0 {
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout.Total)
	}

	initialPrompt, err := standardizePrompt(opts.System, opts.Prompt, opts.Messages, opts.AllowSystemInMessages)
	if err != nil {
		return nil, err
	}
	stopWhen := opts.StopWhen
	if len(stopWhen) == 0 {
		stopWhen = []StopCondition{StepCount(1)}
	}
	maxRetries := 2
	if opts.MaxRetries != nil {
		maxRetries = *opts.MaxRetries
	}

	out := make(chan StreamPart)
	result = &StreamTextResult{Stream: out}
	go streamText(ctx, cancel, opts, initialPrompt, stopWhen, maxRetries, result, out)
	return result, nil
}

func streamText(ctx context.Context, cancel context.CancelFunc, opts StreamTextOptions, initialPrompt standardizedPrompt, stopWhen []StopCondition, maxRetries int, result *StreamTextResult, out chan<- StreamPart) {
	if cancel != nil {
		defer cancel()
	}
	defer close(out)

	responseMessages := []Message{}
	var last *StepResult
	pendingProviderResults := map[string]ToolCall{}
	toolsContext := map[string]any{}

	for {
		select {
		case <-ctx.Done():
			emitAbort(ctx, opts, out, ctx.Err())
			result.Aborted = true
			result.AbortReason = abortReason(ctx.Err())
			return
		default:
		}

		stepCtx := ctx
		var cancelStep context.CancelFunc
		if opts.Timeout.Step > 0 {
			stepCtx, cancelStep = context.WithTimeout(ctx, opts.Timeout.Step)
		}

		stepNumber := len(result.Steps)
		model := opts.Model
		system := opts.System
		stepMessages := append([]Message{}, initialPrompt.Messages...)
		stepTools := opts.Tools
		if len(opts.ActiveTools) > 0 {
			stepTools = FilterActiveTools(stepTools, opts.ActiveTools)
		}
		toolChoice := opts.ToolChoice
		if err := validatePreparedTools(opts.Tools, opts.ActiveTools, toolChoice); err != nil {
			if cancelStep != nil {
				cancelStep()
			}
			sendStreamError(ctx, opts, out, err)
			return
		}
		providerOptions := cloneProviderOptions(opts.ProviderOptions)
		if opts.PrepareStep != nil {
			prepared, err := opts.PrepareStep(PrepareStepOptions{
				Model:        model,
				Steps:        result.Steps,
				StepNumber:   stepNumber,
				Messages:     append(stepMessages, responseMessages...),
				ToolsContext: cloneAnyMap(toolsContext),
			})
			if err != nil {
				if cancelStep != nil {
					cancelStep()
				}
				sendStreamError(ctx, opts, out, err)
				return
			}
			if prepared != nil {
				if prepared.Model != nil {
					model = prepared.Model
				}
				if prepared.System != "" {
					system = prepared.System
				}
				if prepared.Messages != nil {
					stepMessages = prepared.Messages
				}
				if prepared.Tools != nil {
					stepTools = prepared.Tools
				}
				if prepared.ToolChoice.Type != "" {
					toolChoice = prepared.ToolChoice
				}
				providerOptions = mergeProviderOptions(providerOptions, prepared.ProviderOptions)
				toolsContext = mergeToolsContext(toolsContext, prepared.ToolsContext)
			}
		}
		if err := validatePreparedTools(stepTools, nil, toolChoice); err != nil {
			if cancelStep != nil {
				cancelStep()
			}
			sendStreamError(ctx, opts, out, err)
			return
		}

		conversionOptions, err := promptConversionOptionsForModel(stepCtx, model)
		if err != nil {
			if cancelStep != nil {
				cancelStep()
			}
			sendStreamError(ctx, opts, out, err)
			return
		}
		if opts.Download != nil {
			conversionOptions.Download = opts.Download
		}
		promptMessages, err := toLanguageModelPromptWithOptions(stepCtx, standardizedPrompt{
			System:   systemMessages(system, initialPrompt.System),
			Messages: stepMessages,
		}, responseMessages, conversionOptions)
		if err != nil {
			if cancelStep != nil {
				cancelStep()
			}
			sendStreamError(ctx, opts, out, err)
			return
		}

		callOptions := LanguageModelCallOptions{
			Prompt:           promptMessages,
			MaxOutputTokens:  opts.MaxOutputTokens,
			Temperature:      opts.Temperature,
			TopP:             opts.TopP,
			TopK:             opts.TopK,
			PresencePenalty:  opts.PresencePenalty,
			FrequencyPenalty: opts.FrequencyPenalty,
			StopSequences:    opts.StopSequences,
			Seed:             opts.Seed,
			Reasoning:        opts.Reasoning,
			ResponseFormat:   outputResponseFormat(opts.Output, opts.ResponseFormat),
			Tools:            prepareModelTools(stepTools, toolChoice),
			ToolChoice:       normalizeToolChoice(toolChoice),
			ProviderOptions:  providerOptions,
			Headers:          withUserAgent(opts.Headers, "go-ai/"+Version),
		}

		var streamResult *LanguageModelStreamResult
		callID := emitLanguageModelCallStart(ctx, opts.Telemetry, opts.TelemetryOptions, OperationStreamText, model, stepNumber, callOptions)
		err = retry.Do(stepCtx, maxRetries, func() error {
			result, err := model.DoStream(stepCtx, callOptions)
			if err != nil {
				return err
			}
			streamResult = result
			return nil
		})
		if err != nil {
			if cancelStep != nil {
				cancelStep()
			}
			if isAbortError(err) {
				emitAbort(ctx, opts, out, err)
				result.Aborted = true
				result.AbortReason = abortReason(err)
				return
			}
			sendStreamError(ctx, opts, out, err)
			return
		}
		if streamResult == nil {
			if cancelStep != nil {
				cancelStep()
			}
			sendStreamError(ctx, opts, out, &SDKError{Kind: ErrNoOutputGenerated, Message: "model returned nil stream result"})
			return
		}
		if streamResult.Stream == nil {
			if cancelStep != nil {
				cancelStep()
			}
			sendStreamError(ctx, opts, out, &SDKError{Kind: ErrNoOutputGenerated, Message: "model returned nil stream"})
			return
		}

		streamResult.Stream = applyStreamTransforms(stepCtx, streamResult.Stream, opts.Transforms, stepTools, func() {
			if cancelStep != nil {
				cancelStep()
			}
		})
		accumulated := consumeStreamStep(stepCtx, opts, streamResult, system, promptMessages, stepTools, cloneAnyMap(toolsContext), out)
		emitLanguageModelStreamCallEnd(ctx, opts.Telemetry, opts.TelemetryOptions, OperationStreamText, model, stepNumber, callID, streamResult, accumulated)
		if cancelStep != nil {
			cancelStep()
		}
		if accumulated.aborted {
			result.Request = streamResult.Request
			result.Response = streamResult.Response
			result.Aborted = true
			result.AbortReason = accumulated.abortReason
			emitAbort(ctx, opts, out, accumulated.abortErr)
			return
		}
		if accumulated.err != nil {
			result.Request = streamResult.Request
			result.Response = streamResult.Response
			sendStreamError(ctx, opts, out, accumulated.err)
			return
		}

		toolCalls, parsedContent, err := parseToolCalls(stepCtx, accumulated.content, parseToolCallsOptions{Tools: stepTools})
		if err != nil {
			sendStreamError(ctx, opts, out, err)
			return
		}
		accumulated.content = parsedContent
		clientToolResults := accumulated.toolResults
		for _, call := range toolCalls {
			if call.ProviderExecuted {
				if !hasToolResult(accumulated.content, call.ToolCallID) {
					pendingProviderResults[call.ToolCallID] = call
				}
				continue
			}
		}
		for _, part := range accumulated.content {
			if result, ok := part.(ToolResultPart); ok {
				delete(pendingProviderResults, result.ToolCallID)
			}
		}

		responseMessages = append(responseMessages, toResponseMessages(accumulated.content, clientToolResults)...)
		stepResponse := streamResult.Response
		stepResponse.Messages = append([]Message{}, responseMessages...)
		if stepResponse.ID == "" {
			stepResponse.ID = fmt.Sprintf("resp-%d", time.Now().UnixNano())
		}
		if stepResponse.Timestamp.IsZero() {
			stepResponse.Timestamp = time.Now()
		}
		if stepResponse.ModelID == "" {
			stepResponse.ModelID = model.ModelID()
		}

		step := &StepResult{
			CallID:           callID,
			StepNumber:       stepNumber,
			Provider:         model.Provider(),
			ModelID:          model.ModelID(),
			Content:          accumulated.content,
			Text:             TextFromParts(accumulated.content),
			FinishReason:     accumulated.finishReason.Unified,
			RawFinishReason:  accumulated.finishReason.Raw,
			Usage:            accumulated.usage,
			Warnings:         accumulated.warnings,
			ProviderMetadata: accumulated.providerMetadata,
			Request:          streamResult.Request,
			Response:         stepResponse,
			ToolCalls:        toolCalls,
			ToolResults:      clientToolResults,
		}
		LogWarnings(step.Warnings, model.Provider(), model.ModelID())
		result.Steps = append(result.Steps, step)
		last = step
		result.Request = step.Request
		result.Response = step.Response
		result.Text = step.Text
		result.Content = append([]Part{}, step.Content...)
		result.FinishReason = step.FinishReason
		result.RawFinishReason = step.RawFinishReason
		result.Usage = AddUsage(result.Usage, step.Usage)
		result.Warnings = append(result.Warnings, step.Warnings...)
		result.ProviderMetadata = step.ProviderMetadata
		result.ToolCalls = step.ToolCalls
		result.ToolResults = step.ToolResults
		emitStepFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStepFinish, EventStreamTextStepFinish, OperationStreamText, step)

		emitStreamChunk(ctx, opts, out, StreamPart{Type: "finish-step", FinishReason: accumulated.finishReason, Usage: accumulated.usage, Warnings: accumulated.warnings, ProviderMetadata: accumulated.providerMetadata})

		shouldStop, err := stopConditionMet(ctx, stopWhen, result.Steps)
		if err != nil {
			sendStreamError(ctx, opts, out, err)
			return
		}
		clientToolCallCount := 0
		for _, call := range toolCalls {
			if !call.ProviderExecuted {
				clientToolCallCount++
			}
		}
		shouldContinue := ((clientToolCallCount > 0 && len(clientToolResults) == clientToolCallCount) || len(pendingProviderResults) > 0) && !shouldStop
		if !shouldContinue {
			break
		}
	}

	if last == nil {
		sendStreamError(ctx, opts, out, &SDKError{Kind: ErrNoOutputGenerated, Message: "no stream step completed"})
		return
	}
	output, outputGenerated, outputErr := parseCompleteTextOutput(opts.Output, result.Text, result.Response, result.Usage, FinishReason{Unified: result.FinishReason, Raw: result.RawFinishReason})
	result.Output = output
	result.OutputGenerated = outputGenerated
	result.OutputErr = outputErr
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventStreamTextFinish, OperationStreamText, result, map[string]any{
		"finish_reason":     result.FinishReason,
		"raw_finish_reason": result.RawFinishReason,
		"usage":             result.Usage,
		"step_count":        len(result.Steps),
	})
	emitStreamChunk(ctx, opts, out, StreamPart{Type: "finish", FinishReason: FinishReason{Unified: result.FinishReason, Raw: result.RawFinishReason}, Usage: result.Usage, Warnings: result.Warnings, ProviderMetadata: result.ProviderMetadata})
}

type streamStepAccumulation struct {
	content          []Part
	finishReason     FinishReason
	usage            Usage
	warnings         []Warning
	providerMetadata ProviderMetadata
	toolResults      []ToolResultPart
	aborted          bool
	abortErr         error
	abortReason      string
	err              error
}

func consumeStreamStep(ctx context.Context, opts StreamTextOptions, streamResult *LanguageModelStreamResult, system string, promptMessages []Message, tools map[string]Tool, toolsContext map[string]any, out chan<- StreamPart) streamStepAccumulation {
	acc := streamStepAccumulation{finishReason: FinishReason{Unified: FinishUnknown}}
	toolInput := map[string]string{}
	blocked := map[string]bool{}
	var textOutput string
	var lastPartialOutput any
	var haveLastPartialOutput bool
	var publishedElements int

	for {
		var part StreamPart
		var ok bool
		select {
		case <-ctx.Done():
			acc.aborted = true
			acc.abortErr = ctx.Err()
			acc.abortReason = abortReason(ctx.Err())
			return acc
		case part, ok = <-streamResult.Stream:
			if !ok {
				return acc
			}
		}
		if part.Type == "" {
			part.Type = inferStreamPartType(part)
		}
		switch part.Type {
		case "error":
			acc.err = part.Err
			if acc.err == nil {
				acc.err = &SDKError{Kind: ErrNoOutputGenerated, Message: "provider stream error"}
			}
			return acc
		case "stream-start":
			acc.warnings = append(acc.warnings, part.Warnings...)
			emitStreamChunk(ctx, opts, out, part)
		case "response-metadata":
			if part.Response.ID != "" || !part.Response.Timestamp.IsZero() || part.Response.ModelID != "" || len(part.Response.Headers) > 0 || part.Response.Body != nil || len(part.Response.Messages) > 0 {
				streamResult.Response = part.Response
			}
			emitStreamChunk(ctx, opts, out, part)
		case "text-delta":
			acc.content = appendTextDelta(acc.content, part.TextDelta)
			textOutput += part.TextDelta
			partial, err := parsePartialTextOutput(opts.Output, textOutput)
			if err != nil {
				acc.err = err
				return acc
			}
			if partial.OK && (!haveLastPartialOutput || !DeepEqual(lastPartialOutput, partial.Value)) {
				part.PartialOutput = partial.Value
				lastPartialOutput = partial.Value
				haveLastPartialOutput = true
			}
			emitStreamChunk(ctx, opts, out, part)
			if partial.OK {
				var elements []any
				elements, publishedElements = elementsFromPartialOutput(opts.Output, partial.Value, publishedElements)
				for _, element := range elements {
					if !emitStreamChunk(ctx, opts, out, StreamPart{Type: "element", Element: element}) {
						acc.aborted = true
						acc.abortErr = ctx.Err()
						acc.abortReason = abortReason(ctx.Err())
						return acc
					}
				}
			}
		case "reasoning-delta":
			acc.content = appendReasoningDelta(acc.content, part.ReasoningDelta, part.ProviderMetadata)
			emitStreamChunk(ctx, opts, out, part)
		case "file", "source":
			if part.Content != nil {
				acc.content = append(acc.content, part.Content)
			}
			emitStreamChunk(ctx, opts, out, part)
		case "tool-input-delta":
			toolInput[part.ToolCallID] += part.ToolInputDelta
			emitStreamChunk(ctx, opts, out, part)
		case "tool-input-end":
			if part.ToolInput != "" {
				toolInput[part.ToolCallID] = part.ToolInput
			}
			emitStreamChunk(ctx, opts, out, part)
		case "tool-call":
			input := part.ToolInput
			if input == "" {
				input = toolInput[part.ToolCallID]
			}
			toolPart := ToolCallPart{ToolCallID: part.ToolCallID, ToolName: part.ToolName, InputRaw: input, ProviderMetadata: part.ProviderMetadata}
			call, repairedPart, err := parseToolCall(ctx, toolPart, parseToolCallsOptions{
				Tools:          tools,
				RepairToolCall: opts.RepairToolCall,
				System:         system,
				Messages:       promptMessages,
			})
			if err != nil {
				acc.err = err
				return acc
			}
			if repairedPart != nil {
				toolPart = *repairedPart
				input = toolPart.InputJSON()
			}
			acc.content = append(acc.content, toolPart)
			emitStreamChunk(ctx, opts, out, StreamPart{Type: "tool-call", ToolCallID: toolPart.ToolCallID, ToolName: toolPart.ToolName, ToolInput: input, ProviderMetadata: toolPart.ProviderMetadata, Raw: part.Raw})

			tool, ok := tools[call.ToolName]
			if !ok || tool.Execute == nil || call.Invalid {
				blocked[call.ToolCallID] = true
				continue
			}
			if opts.ToolApproval != nil || tool.RequiresApproval || tool.NeedsApproval != nil {
				decision, err := resolveToolApproval(ctx, tools, call, opts.ToolApproval, promptMessages, cloneAnyMap(toolsContext))
				if err != nil {
					acc.err = err
					return acc
				}
				if ApprovalBlocksToolExecution(decision) {
					blocked[call.ToolCallID] = true
					result := ToolResultPart{
						ToolCallID: call.ToolCallID,
						ToolName:   call.ToolName,
						Input:      call.Input,
						Output:     ToolResultOutput{Type: "execution-denied", Reason: decision.Reason},
					}
					acc.content = append(acc.content, result)
					acc.toolResults = append(acc.toolResults, result)
					emitStreamChunk(ctx, opts, out, StreamPart{Type: "tool-result", ToolCallID: result.ToolCallID, ToolName: result.ToolName, Content: result})
				}
			}
			if blocked[call.ToolCallID] {
				continue
			}
			result := executeTool(ctx, opts.Timeout.Tool, call, tool, promptMessages, cloneAnyMap(toolsContext), opts.OnToolExecutionStart, opts.OnToolExecutionEnd)
			acc.content = append(acc.content, result)
			acc.toolResults = append(acc.toolResults, result)
			emitStreamChunk(ctx, opts, out, StreamPart{Type: "tool-result", ToolCallID: result.ToolCallID, ToolName: result.ToolName, Content: result})
		case "finish":
			acc.finishReason = part.FinishReason
			acc.usage = part.Usage
			acc.warnings = append(acc.warnings, part.Warnings...)
			acc.providerMetadata = part.ProviderMetadata
		default:
			if opts.IncludeRawChunks || part.Type != "raw" {
				emitStreamChunk(ctx, opts, out, part)
			}
		}
	}
}

func appendTextDelta(parts []Part, delta string) []Part {
	if delta == "" {
		return parts
	}
	if len(parts) > 0 {
		if text, ok := parts[len(parts)-1].(TextPart); ok {
			text.Text += delta
			parts[len(parts)-1] = text
			return parts
		}
	}
	return append(parts, TextPart{Text: delta})
}

func appendReasoningDelta(parts []Part, delta string, metadata ProviderMetadata) []Part {
	if delta == "" {
		return parts
	}
	if len(parts) > 0 {
		if reasoning, ok := parts[len(parts)-1].(ReasoningPart); ok {
			reasoning.Text += delta
			reasoning.ProviderMetadata = mergeMetadata(reasoning.ProviderMetadata, metadata)
			parts[len(parts)-1] = reasoning
			return parts
		}
	}
	return append(parts, ReasoningPart{Text: delta, ProviderMetadata: metadata})
}

func sendStreamError(ctx context.Context, opts StreamTextOptions, out chan<- StreamPart, err error) {
	emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventStreamTextError, OperationStreamText, err)
	emitStreamChunk(ctx, opts, out, StreamPart{Type: "error", Err: err})
}

func emitAbort(ctx context.Context, opts StreamTextOptions, out chan<- StreamPart, err error) {
	abortCtx := context.WithoutCancel(ctx)
	timeout := opts.Timeout.Chunk
	if timeout <= 0 {
		timeout = 100 * time.Millisecond
	}
	abortCtx, cancel := context.WithTimeout(abortCtx, timeout)
	defer cancel()
	emitStreamChunk(abortCtx, opts, out, StreamPart{Type: "abort", AbortReason: abortReason(err)})
}

func emitStreamChunk(ctx context.Context, opts StreamTextOptions, out chan<- StreamPart, part StreamPart) bool {
	emitChunk(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnChunk, EventStreamTextChunk, OperationStreamText, part)
	select {
	case <-ctx.Done():
		return false
	case out <- part:
		return true
	}
}

func inferStreamPartType(part StreamPart) string {
	switch {
	case part.Err != nil:
		return "error"
	case part.TextDelta != "":
		return "text-delta"
	case part.ReasoningDelta != "":
		return "reasoning-delta"
	case part.ToolInputDelta != "":
		return "tool-input-delta"
	case part.ToolCallID != "" && part.ToolName != "" && part.ToolInput != "":
		return "tool-call"
	case part.Content != nil:
		return part.Content.PartType()
	case part.Element != nil:
		return "element"
	case part.FinishReason.Unified != "" || part.FinishReason.Raw != "" || !usageIsZero(part.Usage):
		return "finish"
	default:
		return ""
	}
}

func isAbortError(err error) bool {
	return errors.Is(err, context.Canceled)
}

func abortReason(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func usageIsZero(usage Usage) bool {
	return usage.InputTokens == nil &&
		usage.OutputTokens == nil &&
		usage.TotalTokens == nil &&
		usage.ReasoningTokens == nil &&
		usage.CachedInputTokens == nil
}

func applyStreamTransforms(ctx context.Context, stream <-chan StreamPart, transforms []StreamTransform, tools map[string]Tool, stopStream func()) <-chan StreamPart {
	if len(transforms) == 0 {
		return stream
	}
	var stopOnce sync.Once
	opts := StreamTransformOptions{
		Tools: tools,
		StopStream: func() {
			stopOnce.Do(func() {
				if stopStream != nil {
					stopStream()
				}
			})
		},
	}
	for _, transform := range transforms {
		if transform == nil {
			continue
		}
		if transformed := transform(ctx, stream, opts); transformed != nil {
			stream = transformed
		}
	}
	return stream
}

func SmoothStream(options ...SmoothStreamOptions) StreamTransform {
	delay := 10 * time.Millisecond
	chunking := SmoothStreamChunkByWord
	var detect ChunkDetector
	if len(options) > 0 {
		if options[0].Delay != nil {
			delay = *options[0].Delay
		}
		if options[0].Chunking != "" {
			chunking = options[0].Chunking
		}
		detect = options[0].DetectChunk
	}
	if detect == nil {
		detect = smoothStreamDetector(chunking)
	}

	return func(ctx context.Context, in <-chan StreamPart, _ StreamTransformOptions) <-chan StreamPart {
		out := make(chan StreamPart)
		go func() {
			defer close(out)

			var buffer strings.Builder
			var bufferedType string
			var bufferedID string
			var bufferedMetadata ProviderMetadata

			flush := func() bool {
				if buffer.Len() == 0 || bufferedType == "" {
					return true
				}
				part := StreamPart{Type: bufferedType, ID: bufferedID, ProviderMetadata: bufferedMetadata}
				setSmoothText(&part, buffer.String())
				buffer.Reset()
				bufferedMetadata = nil
				return sendTransformedPart(ctx, out, part)
			}

			for {
				select {
				case <-ctx.Done():
					return
				case part, ok := <-in:
					if !ok {
						flush()
						return
					}
					if part.Type == "" {
						part.Type = inferStreamPartType(part)
					}
					if !isSmoothableStreamPart(part) {
						if !flush() || !sendTransformedPart(ctx, out, part) {
							return
						}
						continue
					}

					if buffer.Len() > 0 && (part.Type != bufferedType || part.ID != bufferedID) {
						if !flush() {
							return
						}
					}

					buffer.WriteString(smoothPartText(part))
					bufferedType = part.Type
					bufferedID = part.ID
					if len(part.ProviderMetadata) > 0 {
						bufferedMetadata = mergeMetadata(bufferedMetadata, part.ProviderMetadata)
					}

					for {
						chunk, ok, err := detect(buffer.String())
						if err != nil {
							sendTransformedPart(ctx, out, StreamPart{Type: "error", Err: err})
							return
						}
						if !ok {
							break
						}
						buffered := buffer.String()
						if chunk == "" {
							sendTransformedPart(ctx, out, StreamPart{Type: "error", Err: &SDKError{Kind: ErrInvalidArgument, Message: "chunk detector returned an empty chunk"}})
							return
						}
						if !strings.HasPrefix(buffered, chunk) {
							sendTransformedPart(ctx, out, StreamPart{Type: "error", Err: &SDKError{Kind: ErrInvalidArgument, Message: "chunk detector must return a prefix of the buffer"}})
							return
						}

						part := StreamPart{Type: bufferedType, ID: bufferedID}
						setSmoothText(&part, chunk)
						if !sendTransformedPart(ctx, out, part) {
							return
						}
						buffer.Reset()
						buffer.WriteString(buffered[len(chunk):])
						if delay > 0 {
							timer := time.NewTimer(delay)
							select {
							case <-ctx.Done():
								timer.Stop()
								return
							case <-timer.C:
							}
						}
					}
				}
			}
		}()
		return out
	}
}

func smoothStreamDetector(chunking SmoothStreamChunking) ChunkDetector {
	return func(buffer string) (string, bool, error) {
		var loc []int
		switch chunking {
		case "", SmoothStreamChunkByWord:
			loc = smoothStreamWordRE.FindStringIndex(buffer)
		case SmoothStreamChunkByLine:
			loc = smoothStreamLineRE.FindStringIndex(buffer)
		default:
			return "", false, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("unsupported smooth stream chunking: %q", chunking)}
		}
		if loc == nil {
			return "", false, nil
		}
		return buffer[:loc[1]], true, nil
	}
}

func isSmoothableStreamPart(part StreamPart) bool {
	return (part.Type == "text-delta" && part.TextDelta != "") || (part.Type == "reasoning-delta" && part.ReasoningDelta != "")
}

func smoothPartText(part StreamPart) string {
	if part.Type == "reasoning-delta" {
		return part.ReasoningDelta
	}
	return part.TextDelta
}

func setSmoothText(part *StreamPart, text string) {
	if part.Type == "reasoning-delta" {
		part.ReasoningDelta = text
		return
	}
	part.TextDelta = text
}

func sendTransformedPart(ctx context.Context, out chan<- StreamPart, part StreamPart) bool {
	select {
	case <-ctx.Done():
		return false
	case out <- part:
		return true
	}
}
