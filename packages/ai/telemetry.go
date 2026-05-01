package ai

import (
	"context"
	"strings"
	"sync/atomic"
	"time"
)

type Telemetry interface {
	RecordEvent(context.Context, Event)
}

type TelemetryOptions struct {
	IsEnabled       *bool
	RecordInputs    *bool
	RecordOutputs   *bool
	FunctionID      string
	AttributeFilter TelemetryAttributeFilter
}

type TelemetryAttributeFilter func(Event, string, any) bool

var telemetryCallCounter uint64

func emitStart(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, cb func(StartEvent), name, operation string, model interface {
	Provider() string
	ModelID() string
}, attrs map[string]any) {
	provider, modelID := modelInfo(model)
	callID := nextTelemetryCallID()
	event := StartEvent{Operation: operation, CallID: callID, Provider: provider, ModelID: modelID}
	if cb != nil {
		cb(event)
	}
	recordTelemetry(ctx, telemetry, opts, Event{
		Name:       name,
		Operation:  operation,
		CallID:     callID,
		Timestamp:  time.Now(),
		Provider:   provider,
		ModelID:    modelID,
		Attributes: attrs,
	})
}

func emitStepFinish(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, cb func(StepFinishEvent), name, operation string, step *StepResult) {
	callID := ""
	if step != nil {
		callID = step.CallID
	}
	if cb != nil {
		cb(StepFinishEvent{Operation: operation, CallID: callID, Step: step})
	}
	event := Event{
		Name:       name,
		Operation:  operation,
		CallID:     callID,
		Timestamp:  time.Now(),
		Attributes: map[string]any{},
	}
	if step != nil {
		event.StepNumber = &step.StepNumber
		event.Provider = step.Provider
		event.ModelID = step.ModelID
		event.Attributes["finish_reason"] = step.FinishReason
		event.Attributes["raw_finish_reason"] = step.RawFinishReason
		event.Attributes["usage"] = step.Usage
	}
	recordTelemetry(ctx, telemetry, opts, event)
}

func emitChunk(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, cb func(ChunkEvent), name, operation string, chunk StreamPart) {
	if cb != nil {
		cb(ChunkEvent{Operation: operation, Chunk: chunk})
	}
	recordTelemetry(ctx, telemetry, opts, Event{
		Name:       name,
		Operation:  operation,
		Timestamp:  time.Now(),
		Attributes: map[string]any{"chunk_type": chunk.Type},
		Err:        chunk.Err,
	})
}

func emitFinish(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, cb func(FinishEvent), name, operation string, result any, attrs map[string]any) {
	if cb != nil {
		cb(FinishEvent{Operation: operation, Result: result})
	}
	recordTelemetry(ctx, telemetry, opts, Event{
		Name:       name,
		Operation:  operation,
		Timestamp:  time.Now(),
		Attributes: attrs,
	})
}

func emitError(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, cb func(ErrorEvent), name, operation string, err error) {
	if err == nil {
		return
	}
	if cb != nil {
		cb(ErrorEvent{Operation: operation, Err: err})
	}
	recordTelemetry(ctx, telemetry, opts, Event{
		Name:      name,
		Operation: operation,
		Timestamp: time.Now(),
		Err:       err,
	})
}

func emitLanguageModelCallStart(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, operation string, model interface {
	Provider() string
	ModelID() string
}, stepNumber int, callOptions LanguageModelCallOptions) string {
	provider, modelID := modelInfo(model)
	callID := nextTelemetryCallID()
	attrs := map[string]any{
		"input.prompt":           append([]Message(nil), callOptions.Prompt...),
		"input.tools":            append([]ModelTool(nil), callOptions.Tools...),
		"input.tool_choice":      callOptions.ToolChoice,
		"input.provider_options": cloneProviderOptions(callOptions.ProviderOptions),
	}
	recordTelemetry(ctx, telemetry, opts, Event{
		Name:       EventOnLanguageModelCallStart,
		Operation:  operation,
		CallID:     callID,
		StepNumber: &stepNumber,
		Timestamp:  time.Now(),
		Provider:   provider,
		ModelID:    modelID,
		Attributes: attrs,
	})
	return callID
}

func emitLanguageModelCallEnd(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, operation string, model interface {
	Provider() string
	ModelID() string
}, stepNumber int, callID string, result *LanguageModelGenerateResult) {
	provider, modelID := modelInfo(model)
	attrs := map[string]any{}
	if result != nil {
		attrs["finish_reason"] = result.FinishReason.Unified
		attrs["raw_finish_reason"] = result.FinishReason.Raw
		attrs["usage"] = result.Usage
		attrs["output.content"] = append([]Part(nil), result.Content...)
		attrs["output.response_id"] = result.Response.ID
	}
	recordTelemetry(ctx, telemetry, opts, Event{
		Name:       EventOnLanguageModelCallEnd,
		Operation:  operation,
		CallID:     callID,
		StepNumber: &stepNumber,
		Timestamp:  time.Now(),
		Provider:   provider,
		ModelID:    modelID,
		Attributes: attrs,
	})
}

func emitLanguageModelStreamCallEnd(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, operation string, model interface {
	Provider() string
	ModelID() string
}, stepNumber int, callID string, result *LanguageModelStreamResult, acc streamStepAccumulation) {
	provider, modelID := modelInfo(model)
	attrs := map[string]any{
		"finish_reason":     acc.finishReason.Unified,
		"raw_finish_reason": acc.finishReason.Raw,
		"usage":             acc.usage,
		"output.content":    append([]Part(nil), acc.content...),
	}
	if result != nil {
		attrs["output.response_id"] = result.Response.ID
	}
	recordTelemetry(ctx, telemetry, opts, Event{
		Name:       EventOnLanguageModelCallEnd,
		Operation:  operation,
		CallID:     callID,
		StepNumber: &stepNumber,
		Timestamp:  time.Now(),
		Provider:   provider,
		ModelID:    modelID,
		Attributes: attrs,
		Err:        acc.err,
	})
}

func recordTelemetry(ctx context.Context, telemetry Telemetry, opts TelemetryOptions, event Event) {
	if opts.IsEnabled != nil && !*opts.IsEnabled {
		return
	}
	if telemetry != nil {
		event = applyTelemetryOptions(event, opts)
		telemetry.RecordEvent(ctx, event)
	}
}

func modelInfo(model interface {
	Provider() string
	ModelID() string
}) (string, string) {
	if model == nil {
		return "", ""
	}
	return model.Provider(), model.ModelID()
}

func applyTelemetryOptions(event Event, opts TelemetryOptions) Event {
	if event.OperationID == "" {
		event.OperationID = diagnosticOperationID(event.Operation)
	}
	event.RecordInputs = opts.RecordInputs
	event.RecordOutputs = opts.RecordOutputs
	event.FunctionID = opts.FunctionID
	event.Attributes = filterTelemetryAttributes(event, opts)
	return event
}

func filterTelemetryAttributes(event Event, opts TelemetryOptions) map[string]any {
	if len(event.Attributes) == 0 {
		return event.Attributes
	}
	recordInputs := true
	if opts.RecordInputs != nil {
		recordInputs = *opts.RecordInputs
	}
	recordOutputs := true
	if opts.RecordOutputs != nil {
		recordOutputs = *opts.RecordOutputs
	}
	filtered := make(map[string]any, len(event.Attributes))
	for key, value := range event.Attributes {
		if !recordInputs && isInputTelemetryAttribute(key) {
			continue
		}
		if !recordOutputs && isOutputTelemetryAttribute(key) {
			continue
		}
		if opts.AttributeFilter != nil && !opts.AttributeFilter(event, key, value) {
			continue
		}
		filtered[key] = value
	}
	return filtered
}

func isInputTelemetryAttribute(key string) bool {
	switch key {
	case "prompt", "query", "text", "filename", "display_title":
		return true
	}
	return strings.HasPrefix(key, "input.")
}

func isOutputTelemetryAttribute(key string) bool {
	return strings.HasPrefix(key, "output.")
}

func nextTelemetryCallID() string {
	id := atomic.AddUint64(&telemetryCallCounter, 1)
	return "call-" + strconvFormatUint(id)
}

func diagnosticOperationID(operation string) string {
	switch operation {
	case OperationGenerateText:
		return OperationIDGenerateText
	case OperationStreamText:
		return OperationIDStreamText
	case OperationEmbed:
		return OperationIDEmbed
	case OperationEmbedMany:
		return OperationIDEmbedMany
	case OperationGenerateObject:
		return OperationIDGenerateObject
	case OperationGenerateImage:
		return OperationIDGenerateImage
	case OperationGenerateVideo:
		return OperationIDGenerateVideo
	case OperationGenerateSpeech:
		return OperationIDGenerateSpeech
	case OperationTranscribe:
		return OperationIDTranscribe
	case OperationRerank:
		return OperationIDRerank
	case OperationUploadFile:
		return OperationIDUploadFile
	case OperationUploadSkill:
		return OperationIDUploadSkill
	default:
		return operation
	}
}

func strconvFormatUint(v uint64) string {
	const digits = "0123456789"
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = digits[v%10]
		v /= 10
	}
	return string(buf[i:])
}
