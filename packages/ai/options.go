package ai

import (
	"context"
	"time"
)

type GenerateTextOptions struct {
	Model                 LanguageModel
	System                string
	Prompt                string
	Messages              []Message
	AllowSystemInMessages bool
	Tools                 map[string]Tool
	ActiveTools           []string
	ToolChoice            ToolChoice
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
	RepairToolCall        ToolCallRepairFunc
	Telemetry             Telemetry
	TelemetryOptions      TelemetryOptions
	OnStart               func(StartEvent)
	OnToolExecutionStart  func(ToolExecutionStartEvent)
	OnToolExecutionEnd    func(ToolExecutionEndEvent)
	OnStepFinish          func(StepFinishEvent)
	OnFinish              func(FinishEvent)
	OnError               func(ErrorEvent)
}

type StreamTextOptions struct {
	GenerateTextOptions
	IncludeRawChunks bool
	OnChunk          func(ChunkEvent)
	Transforms       []StreamTransform
}

type StreamTransform func(context.Context, <-chan StreamPart, StreamTransformOptions) <-chan StreamPart

type StreamTransformOptions struct {
	Tools      map[string]Tool
	StopStream func()
}

type ChunkDetector func(buffer string) (chunk string, ok bool, err error)

type SmoothStreamChunking string

const (
	SmoothStreamChunkByWord SmoothStreamChunking = "word"
	SmoothStreamChunkByLine SmoothStreamChunking = "line"
)

type SmoothStreamOptions struct {
	Delay       *time.Duration
	Chunking    SmoothStreamChunking
	DetectChunk ChunkDetector
}

type TimeoutConfig struct {
	Total time.Duration
	Step  time.Duration
	Tool  time.Duration
	Chunk time.Duration
}

type PrepareStepOptions struct {
	Model        LanguageModel
	Steps        []*StepResult
	StepNumber   int
	Messages     []Message
	ToolsContext map[string]any
}

type PrepareStepResult struct {
	Model           LanguageModel
	System          string
	Messages        []Message
	Tools           map[string]Tool
	ToolChoice      ToolChoice
	ProviderOptions ProviderOptions
	ToolsContext    map[string]any
}

type GenerateTextResult struct {
	Text             string
	Output           any
	OutputGenerated  bool
	OutputErr        error
	Content          []Part
	FinishReason     string
	RawFinishReason  string
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
	Steps            []*StepResult
	ToolCalls        []ToolCall
	ToolResults      []ToolResultPart
}

type StreamTextResult struct {
	Stream           <-chan StreamPart
	Text             string
	Output           any
	OutputGenerated  bool
	OutputErr        error
	Content          []Part
	FinishReason     string
	RawFinishReason  string
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
	Steps            []*StepResult
	ToolCalls        []ToolCall
	ToolResults      []ToolResultPart
}

type GenerateObjectOptions struct {
	Model                 LanguageModel
	Output                string
	Mode                  string
	Schema                any
	SchemaName            string
	SchemaDescription     string
	Enum                  []string
	System                string
	Prompt                string
	Messages              []Message
	AllowSystemInMessages bool
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
	RepairText            func(RepairTextOptions) (string, error)
	Telemetry             Telemetry
	TelemetryOptions      TelemetryOptions
	OnStart               func(StartEvent)
	OnFinish              func(FinishEvent)
	OnError               func(ErrorEvent)
}

type StreamObjectOptions struct {
	GenerateObjectOptions
	IncludeRawChunks bool
}

type RepairTextOptions struct {
	Text  string
	Error error
}

type ToolCallRepairFunc func(context.Context, ToolCallRepairOptions) (*ToolCallPart, error)

type ToolCallRepairOptions struct {
	System      string
	Messages    []Message
	ToolCall    ToolCallPart
	Tools       map[string]Tool
	InputSchema func(toolName string) (any, bool)
	Error       error
}

type GenerateObjectResult struct {
	Object           any
	FinishReason     string
	RawFinishReason  string
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
	Reasoning        string
	Text             string
}

type StreamObjectResult struct {
	Stream   <-chan ObjectStreamPart
	Request  RequestMetadata
	Response ResponseMetadata
}

type ObjectStreamPart struct {
	Type             string
	TextDelta        string
	Object           any
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Raw              any
	Err              error
}

type EmbedOptions struct {
	Model            EmbeddingModel
	Value            string
	MaxRetries       *int
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type EmbedManyOptions struct {
	Model            EmbeddingModel
	Values           []string
	MaxRetries       *int
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	MaxParallelCalls int
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type EmbedResult struct {
	Value            string
	Embedding        []float64
	Usage            EmbeddingUsage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

type EmbedManyResult struct {
	Values           []string
	Embeddings       [][]float64
	Usage            EmbeddingUsage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Responses        []ResponseMetadata
}

type StepResult struct {
	CallID           string
	StepNumber       int
	Provider         string
	ModelID          string
	Content          []Part
	Text             string
	FinishReason     string
	RawFinishReason  string
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
	ToolCalls        []ToolCall
	ToolResults      []ToolResultPart
}
