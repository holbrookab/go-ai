package ai

import "context"

const AgentVersion = "agent-v1"

type Agent interface {
	Version() string
	ID() string
	Tools() map[string]Tool
	Generate(context.Context, AgentCallOptions) (*GenerateTextResult, error)
	Stream(context.Context, AgentStreamOptions) (*StreamTextResult, error)
}

type AgentCallOptions struct {
	Prompt                string
	Messages              []Message
	Options               any
	AllowSystemInMessages bool
	Tools                 map[string]Tool
	ActiveTools           []string
	ToolChoice            ToolChoice
	ToolExecution         ToolExecutionMode
	ToolApproval          *ToolApprovalConfiguration
	StopWhen              []StopCondition
	MaxRetries            *int
	Timeout               TimeoutConfig
	Headers               map[string]string
	ProviderOptions       ProviderOptions
	MaxOutputTokens       *int
	Temperature           *float64
	TopP                  *float64
	TopK                  *float64
	PresencePenalty       *float64
	FrequencyPenalty      *float64
	StopSequences         []string
	Seed                  *int
	Reasoning             string
	Download              DownloadFunction
	Output                *OutputStrategy
	ResponseFormat        *ResponseFormat
	PrepareStep           func(PrepareStepOptions) (*PrepareStepResult, error)
	Telemetry             Telemetry
	OnStart               func(StartEvent)
	OnToolExecutionStart  func(ToolExecutionStartEvent)
	OnToolExecutionEnd    func(ToolExecutionEndEvent)
	OnStepFinish          func(StepFinishEvent)
	OnFinish              func(FinishEvent)
	OnError               func(ErrorEvent)
}

type AgentStreamOptions struct {
	AgentCallOptions
	IncludeRawChunks bool
	OnChunk          func(ChunkEvent)
	Transforms       []StreamTransform
}

type ToolLoopAgentSettings struct {
	ID                   string
	Instructions         string
	Model                LanguageModel
	Tools                map[string]Tool
	ActiveTools          []string
	ToolChoice           ToolChoice
	ToolExecution        ToolExecutionMode
	ToolApproval         *ToolApprovalConfiguration
	StopWhen             []StopCondition
	MaxRetries           *int
	Timeout              TimeoutConfig
	Headers              map[string]string
	ProviderOptions      ProviderOptions
	MaxOutputTokens      *int
	Temperature          *float64
	TopP                 *float64
	TopK                 *float64
	PresencePenalty      *float64
	FrequencyPenalty     *float64
	StopSequences        []string
	Seed                 *int
	Reasoning            string
	Download             DownloadFunction
	Output               *OutputStrategy
	ResponseFormat       *ResponseFormat
	PrepareStep          func(PrepareStepOptions) (*PrepareStepResult, error)
	Telemetry            Telemetry
	Transforms           []StreamTransform
	OnStart              func(StartEvent)
	OnToolExecutionStart func(ToolExecutionStartEvent)
	OnToolExecutionEnd   func(ToolExecutionEndEvent)
	OnStepFinish         func(StepFinishEvent)
	OnFinish             func(FinishEvent)
	OnError              func(ErrorEvent)
	PrepareCall          func(AgentPrepareCallOptions) (*AgentPreparedCall, error)
}

type AgentPrepareCallOptions struct {
	Settings ToolLoopAgentSettings
	Call     AgentCallOptions
}

type AgentPreparedCall struct {
	System                *string
	Prompt                *string
	Messages              []Message
	AllowSystemInMessages *bool
	Model                 LanguageModel
	Tools                 map[string]Tool
	ActiveTools           []string
	ToolChoice            *ToolChoice
	ToolExecution         *ToolExecutionMode
	ToolApproval          *ToolApprovalConfiguration
	StopWhen              []StopCondition
	MaxRetries            *int
	Timeout               *TimeoutConfig
	Headers               map[string]string
	ProviderOptions       ProviderOptions
	MaxOutputTokens       *int
	Temperature           *float64
	TopP                  *float64
	TopK                  *float64
	PresencePenalty       *float64
	FrequencyPenalty      *float64
	StopSequences         []string
	Seed                  *int
	Reasoning             *string
	Download              DownloadFunction
	Output                *OutputStrategy
	ResponseFormat        *ResponseFormat
	PrepareStep           func(PrepareStepOptions) (*PrepareStepResult, error)
	Telemetry             Telemetry
	Transforms            []StreamTransform
}

type ToolLoopAgent struct {
	settings ToolLoopAgentSettings
}

func NewToolLoopAgent(settings ToolLoopAgentSettings) *ToolLoopAgent {
	return &ToolLoopAgent{settings: settings}
}

func (a *ToolLoopAgent) Version() string {
	return AgentVersion
}

func (a *ToolLoopAgent) ID() string {
	if a == nil {
		return ""
	}
	return a.settings.ID
}

func (a *ToolLoopAgent) Tools() map[string]Tool {
	if a == nil {
		return nil
	}
	return a.settings.Tools
}

func (a *ToolLoopAgent) Generate(ctx context.Context, opts AgentCallOptions) (*GenerateTextResult, error) {
	call, err := a.prepareCall(opts)
	if err != nil {
		return nil, err
	}
	return GenerateText(ctx, GenerateTextOptions{
		Model:                 call.Model,
		System:                call.System,
		Prompt:                call.Prompt,
		Messages:              call.Messages,
		AllowSystemInMessages: call.AllowSystemInMessages,
		Tools:                 call.Tools,
		ActiveTools:           opts.ActiveTools,
		ToolChoice:            call.ToolChoice,
		ToolExecution:         call.ToolExecution,
		ToolApproval:          call.ToolApproval,
		StopWhen:              call.StopWhen,
		MaxRetries:            call.MaxRetries,
		Timeout:               call.Timeout,
		Headers:               call.Headers,
		ProviderOptions:       call.ProviderOptions,
		MaxOutputTokens:       call.MaxOutputTokens,
		Temperature:           call.Temperature,
		TopP:                  call.TopP,
		TopK:                  call.TopK,
		PresencePenalty:       call.PresencePenalty,
		FrequencyPenalty:      call.FrequencyPenalty,
		StopSequences:         call.StopSequences,
		Seed:                  call.Seed,
		Reasoning:             call.Reasoning,
		Download:              call.Download,
		Output:                call.Output,
		ResponseFormat:        call.ResponseFormat,
		PrepareStep:           call.PrepareStep,
		Telemetry:             call.Telemetry,
		OnStart:               mergeStartCallbacks(a.settings.OnStart, opts.OnStart),
		OnToolExecutionStart:  mergeToolExecutionStartCallbacks(a.settings.OnToolExecutionStart, opts.OnToolExecutionStart),
		OnToolExecutionEnd:    mergeToolExecutionEndCallbacks(a.settings.OnToolExecutionEnd, opts.OnToolExecutionEnd),
		OnStepFinish:          mergeStepFinishCallbacks(a.settings.OnStepFinish, opts.OnStepFinish),
		OnFinish:              mergeFinishCallbacks(a.settings.OnFinish, opts.OnFinish),
		OnError:               mergeErrorCallbacks(a.settings.OnError, opts.OnError),
	})
}

func (a *ToolLoopAgent) Stream(ctx context.Context, opts AgentStreamOptions) (*StreamTextResult, error) {
	call, err := a.prepareCall(opts.AgentCallOptions)
	if err != nil {
		return nil, err
	}
	return StreamText(ctx, StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:                 call.Model,
			System:                call.System,
			Prompt:                call.Prompt,
			Messages:              call.Messages,
			AllowSystemInMessages: call.AllowSystemInMessages,
			Tools:                 call.Tools,
			ActiveTools:           opts.ActiveTools,
			ToolChoice:            call.ToolChoice,
			ToolExecution:         call.ToolExecution,
			ToolApproval:          call.ToolApproval,
			StopWhen:              call.StopWhen,
			MaxRetries:            call.MaxRetries,
			Timeout:               call.Timeout,
			Headers:               call.Headers,
			ProviderOptions:       call.ProviderOptions,
			MaxOutputTokens:       call.MaxOutputTokens,
			Temperature:           call.Temperature,
			TopP:                  call.TopP,
			TopK:                  call.TopK,
			PresencePenalty:       call.PresencePenalty,
			FrequencyPenalty:      call.FrequencyPenalty,
			StopSequences:         call.StopSequences,
			Seed:                  call.Seed,
			Reasoning:             call.Reasoning,
			Download:              call.Download,
			Output:                call.Output,
			ResponseFormat:        call.ResponseFormat,
			PrepareStep:           call.PrepareStep,
			Telemetry:             call.Telemetry,
			OnStart:               mergeStartCallbacks(a.settings.OnStart, opts.OnStart),
			OnToolExecutionStart:  mergeToolExecutionStartCallbacks(a.settings.OnToolExecutionStart, opts.OnToolExecutionStart),
			OnToolExecutionEnd:    mergeToolExecutionEndCallbacks(a.settings.OnToolExecutionEnd, opts.OnToolExecutionEnd),
			OnStepFinish:          mergeStepFinishCallbacks(a.settings.OnStepFinish, opts.OnStepFinish),
			OnFinish:              mergeFinishCallbacks(a.settings.OnFinish, opts.OnFinish),
			OnError:               mergeErrorCallbacks(a.settings.OnError, opts.OnError),
		},
		IncludeRawChunks: opts.IncludeRawChunks,
		OnChunk:          opts.OnChunk,
		Transforms:       firstStreamTransforms(opts.Transforms, call.Transforms),
	})
}

type preparedAgentCall struct {
	System                string
	Prompt                string
	Messages              []Message
	AllowSystemInMessages bool
	Model                 LanguageModel
	Tools                 map[string]Tool
	ToolChoice            ToolChoice
	ToolExecution         ToolExecutionMode
	ToolApproval          *ToolApprovalConfiguration
	StopWhen              []StopCondition
	MaxRetries            *int
	Timeout               TimeoutConfig
	Headers               map[string]string
	ProviderOptions       ProviderOptions
	MaxOutputTokens       *int
	Temperature           *float64
	TopP                  *float64
	TopK                  *float64
	PresencePenalty       *float64
	FrequencyPenalty      *float64
	StopSequences         []string
	Seed                  *int
	Reasoning             string
	Download              DownloadFunction
	Output                *OutputStrategy
	ResponseFormat        *ResponseFormat
	PrepareStep           func(PrepareStepOptions) (*PrepareStepResult, error)
	Telemetry             Telemetry
	Transforms            []StreamTransform
}

func (a *ToolLoopAgent) prepareCall(opts AgentCallOptions) (preparedAgentCall, error) {
	if a == nil {
		return preparedAgentCall{}, &SDKError{Kind: ErrInvalidArgument, Message: "agent is required"}
	}
	settings := a.settings
	call := preparedAgentCall{
		System:                settings.Instructions,
		Prompt:                opts.Prompt,
		Messages:              opts.Messages,
		AllowSystemInMessages: opts.AllowSystemInMessages,
		Model:                 firstLanguageModel(optsModel(opts), settings.Model),
		Tools:                 firstTools(opts.Tools, settings.Tools),
		ToolChoice:            firstToolChoice(opts.ToolChoice, settings.ToolChoice),
		ToolExecution:         firstToolExecution(opts.ToolExecution, settings.ToolExecution),
		ToolApproval:          firstToolApproval(opts.ToolApproval, settings.ToolApproval),
		StopWhen:              firstStopConditions(opts.StopWhen, settings.StopWhen, []StopCondition{StepCount(20)}),
		MaxRetries:            firstIntPtr(opts.MaxRetries, settings.MaxRetries),
		Timeout:               firstTimeout(opts.Timeout, settings.Timeout),
		Headers:               mergeStringMaps(settings.Headers, opts.Headers),
		ProviderOptions:       mergeProviderOptions(cloneProviderOptions(settings.ProviderOptions), opts.ProviderOptions),
		MaxOutputTokens:       firstIntPtr(opts.MaxOutputTokens, settings.MaxOutputTokens),
		Temperature:           firstFloatPtr(opts.Temperature, settings.Temperature),
		TopP:                  firstFloatPtr(opts.TopP, settings.TopP),
		TopK:                  firstFloatPtr(opts.TopK, settings.TopK),
		PresencePenalty:       firstFloatPtr(opts.PresencePenalty, settings.PresencePenalty),
		FrequencyPenalty:      firstFloatPtr(opts.FrequencyPenalty, settings.FrequencyPenalty),
		StopSequences:         firstStrings(opts.StopSequences, settings.StopSequences),
		Seed:                  firstIntPtr(opts.Seed, settings.Seed),
		Reasoning:             firstString(opts.Reasoning, settings.Reasoning),
		Download:              firstDownload(opts.Download, settings.Download),
		Output:                firstOutputStrategy(opts.Output, settings.Output),
		ResponseFormat:        firstResponseFormat(opts.ResponseFormat, settings.ResponseFormat),
		PrepareStep:           firstPrepareStep(opts.PrepareStep, settings.PrepareStep),
		Telemetry:             firstTelemetry(opts.Telemetry, settings.Telemetry),
		Transforms:            settings.Transforms,
	}
	activeTools := settings.ActiveTools
	if len(opts.Tools) > 0 {
		activeTools = nil
	}
	if settings.PrepareCall != nil {
		prepared, err := settings.PrepareCall(AgentPrepareCallOptions{Settings: settings, Call: opts})
		if err != nil {
			return preparedAgentCall{}, err
		}
		applyPreparedAgentCall(&call, prepared)
		if prepared != nil && prepared.ActiveTools != nil {
			activeTools = prepared.ActiveTools
		}
	}
	if len(activeTools) > 0 {
		call.Tools = FilterActiveTools(call.Tools, activeTools)
	}
	return call, nil
}

func optsModel(AgentCallOptions) LanguageModel { return nil }

func applyPreparedAgentCall(call *preparedAgentCall, prepared *AgentPreparedCall) {
	if prepared == nil {
		return
	}
	if prepared.System != nil {
		call.System = *prepared.System
	}
	if prepared.Prompt != nil {
		call.Prompt = *prepared.Prompt
		call.Messages = nil
	}
	if prepared.Messages != nil {
		call.Messages = prepared.Messages
		call.Prompt = ""
	}
	if prepared.AllowSystemInMessages != nil {
		call.AllowSystemInMessages = *prepared.AllowSystemInMessages
	}
	if prepared.Model != nil {
		call.Model = prepared.Model
	}
	if prepared.Tools != nil {
		call.Tools = prepared.Tools
	}
	if prepared.ToolChoice != nil {
		call.ToolChoice = *prepared.ToolChoice
	}
	if prepared.ToolExecution != nil {
		call.ToolExecution = *prepared.ToolExecution
	}
	if prepared.ToolApproval != nil {
		call.ToolApproval = prepared.ToolApproval
	}
	if prepared.StopWhen != nil {
		call.StopWhen = prepared.StopWhen
	}
	if prepared.MaxRetries != nil {
		call.MaxRetries = prepared.MaxRetries
	}
	if prepared.Timeout != nil {
		call.Timeout = *prepared.Timeout
	}
	call.Headers = mergeStringMaps(call.Headers, prepared.Headers)
	call.ProviderOptions = mergeProviderOptions(call.ProviderOptions, prepared.ProviderOptions)
	if prepared.MaxOutputTokens != nil {
		call.MaxOutputTokens = prepared.MaxOutputTokens
	}
	if prepared.Temperature != nil {
		call.Temperature = prepared.Temperature
	}
	if prepared.TopP != nil {
		call.TopP = prepared.TopP
	}
	if prepared.TopK != nil {
		call.TopK = prepared.TopK
	}
	if prepared.PresencePenalty != nil {
		call.PresencePenalty = prepared.PresencePenalty
	}
	if prepared.FrequencyPenalty != nil {
		call.FrequencyPenalty = prepared.FrequencyPenalty
	}
	if prepared.StopSequences != nil {
		call.StopSequences = prepared.StopSequences
	}
	if prepared.Seed != nil {
		call.Seed = prepared.Seed
	}
	if prepared.Reasoning != nil {
		call.Reasoning = *prepared.Reasoning
	}
	if prepared.Download != nil {
		call.Download = prepared.Download
	}
	if prepared.Output != nil {
		call.Output = prepared.Output
	}
	if prepared.ResponseFormat != nil {
		call.ResponseFormat = prepared.ResponseFormat
	}
	if prepared.PrepareStep != nil {
		call.PrepareStep = prepared.PrepareStep
	}
	if prepared.Telemetry != nil {
		call.Telemetry = prepared.Telemetry
	}
	if prepared.Transforms != nil {
		call.Transforms = prepared.Transforms
	}
}

func firstLanguageModel(values ...LanguageModel) LanguageModel {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstTools(values ...map[string]Tool) map[string]Tool {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstToolChoice(values ...ToolChoice) ToolChoice {
	for _, value := range values {
		if value.Type != "" {
			return value
		}
	}
	return ToolChoice{}
}

func firstToolExecution(values ...ToolExecutionMode) ToolExecutionMode {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func firstToolApproval(values ...*ToolApprovalConfiguration) *ToolApprovalConfiguration {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstStopConditions(values ...[]StopCondition) []StopCondition {
	for _, value := range values {
		if len(value) > 0 {
			return value
		}
	}
	return nil
}

func firstIntPtr(values ...*int) *int {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstFloatPtr(values ...*float64) *float64 {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstTimeout(values ...TimeoutConfig) TimeoutConfig {
	for _, value := range values {
		if value.Total != 0 || value.Step != 0 || value.Tool != 0 || value.Chunk != 0 {
			return value
		}
	}
	return TimeoutConfig{}
}

func mergeStringMaps(base, override map[string]string) map[string]string {
	if len(base) == 0 && len(override) == 0 {
		return nil
	}
	out := map[string]string{}
	for k, v := range base {
		out[k] = v
	}
	for k, v := range override {
		out[k] = v
	}
	return out
}

func firstStrings(values ...[]string) []string {
	for _, value := range values {
		if len(value) > 0 {
			return value
		}
	}
	return nil
}

func firstString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func firstResponseFormat(values ...*ResponseFormat) *ResponseFormat {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstOutputStrategy(values ...*OutputStrategy) *OutputStrategy {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstDownload(values ...DownloadFunction) DownloadFunction {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstPrepareStep(values ...func(PrepareStepOptions) (*PrepareStepResult, error)) func(PrepareStepOptions) (*PrepareStepResult, error) {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstTelemetry(values ...Telemetry) Telemetry {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstStreamTransforms(values ...[]StreamTransform) []StreamTransform {
	for _, value := range values {
		if len(value) > 0 {
			return value
		}
	}
	return nil
}

func mergeStartCallbacks(callbacks ...func(StartEvent)) func(StartEvent) {
	return func(event StartEvent) {
		for _, callback := range callbacks {
			if callback != nil {
				callback(event)
			}
		}
	}
}

func mergeToolExecutionStartCallbacks(callbacks ...func(ToolExecutionStartEvent)) func(ToolExecutionStartEvent) {
	return func(event ToolExecutionStartEvent) {
		for _, callback := range callbacks {
			if callback != nil {
				callback(event)
			}
		}
	}
}

func mergeToolExecutionEndCallbacks(callbacks ...func(ToolExecutionEndEvent)) func(ToolExecutionEndEvent) {
	return func(event ToolExecutionEndEvent) {
		for _, callback := range callbacks {
			if callback != nil {
				callback(event)
			}
		}
	}
}

func mergeStepFinishCallbacks(callbacks ...func(StepFinishEvent)) func(StepFinishEvent) {
	return func(event StepFinishEvent) {
		for _, callback := range callbacks {
			if callback != nil {
				callback(event)
			}
		}
	}
}

func mergeFinishCallbacks(callbacks ...func(FinishEvent)) func(FinishEvent) {
	return func(event FinishEvent) {
		for _, callback := range callbacks {
			if callback != nil {
				callback(event)
			}
		}
	}
}

func mergeErrorCallbacks(callbacks ...func(ErrorEvent)) func(ErrorEvent) {
	return func(event ErrorEvent) {
		for _, callback := range callbacks {
			if callback != nil {
				callback(event)
			}
		}
	}
}
