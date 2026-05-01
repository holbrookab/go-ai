package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

const Version = "0.2.0"

type ProviderOptions map[string]any
type ProviderMetadata map[string]any
type JSONValue any
type JSONObject map[string]JSONValue
type JSONArray []JSONValue
type JSONSchema map[string]any

type Warning struct {
	Type    string
	Feature string
	Setting string
	Message string
	Details string
}

type Usage struct {
	InputTokens       *int
	OutputTokens      *int
	TotalTokens       *int
	ReasoningTokens   *int
	CachedInputTokens *int
}

func AddUsage(a, b Usage) Usage {
	return Usage{
		InputTokens:       addIntPtr(a.InputTokens, b.InputTokens),
		OutputTokens:      addIntPtr(a.OutputTokens, b.OutputTokens),
		TotalTokens:       addIntPtr(a.TotalTokens, b.TotalTokens),
		ReasoningTokens:   addIntPtr(a.ReasoningTokens, b.ReasoningTokens),
		CachedInputTokens: addIntPtr(a.CachedInputTokens, b.CachedInputTokens),
	}
}

func addIntPtr(a, b *int) *int {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	v := *a + *b
	return &v
}

type FinishReason struct {
	Unified string
	Raw     string
}

const (
	FinishStop          = "stop"
	FinishLength        = "length"
	FinishToolCalls     = "tool-calls"
	FinishContentFilter = "content-filter"
	FinishError         = "error"
	FinishOther         = "other"
	FinishUnknown       = "unknown"
)

type ResponseMetadata struct {
	ID         string
	Timestamp  time.Time
	ModelID    string
	StatusCode int
	StatusText string
	Headers    map[string]string
	Body       any
	Messages   []Message
}

type RequestMetadata struct {
	Method  string
	URL     string
	Headers map[string]string
	Body    any
}

type Provider interface {
	LanguageModel(modelID string) LanguageModel
}

type EmbeddingProvider interface {
	EmbeddingModel(modelID string) EmbeddingModel
}

type ImageProvider interface {
	ImageModel(modelID string) ImageModel
}

type VideoProvider interface {
	VideoModel(modelID string) VideoModel
}

type SpeechProvider interface {
	SpeechModel(modelID string) SpeechModel
}

type TranscriptionProvider interface {
	TranscriptionModel(modelID string) TranscriptionModel
}

type RerankingProvider interface {
	RerankingModel(modelID string) RerankingModel
}

type LanguageModel interface {
	Provider() string
	ModelID() string
	SupportedURLs(ctx context.Context) (map[string][]string, error)
	DoGenerate(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error)
	DoStream(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelStreamResult, error)
}

type LanguageModelCallOptions struct {
	Prompt           []Message
	MaxOutputTokens  *int
	Temperature      *float64
	TopP             *float64
	TopK             *float64
	PresencePenalty  *float64
	FrequencyPenalty *float64
	StopSequences    []string
	Seed             *int
	Reasoning        string
	ResponseFormat   *ResponseFormat
	Tools            []ModelTool
	ToolChoice       ToolChoice
	ProviderOptions  ProviderOptions
	Headers          map[string]string
}

type ResponseFormat struct {
	Type        string
	Schema      any
	Name        string
	Description string
}

type LanguageModelGenerateResult struct {
	Content          []Part
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Request          RequestMetadata
	Response         ResponseMetadata
}

type LanguageModelStreamResult struct {
	Stream   <-chan StreamPart
	Request  RequestMetadata
	Response ResponseMetadata
}

type EmbeddingModel interface {
	Provider() string
	ModelID() string
	MaxEmbeddingsPerCall() int
	DoEmbed(ctx context.Context, opts EmbeddingModelCallOptions) (*EmbeddingModelResult, error)
}

type ImageModel interface {
	Provider() string
	ModelID() string
	DoGenerateImage(ctx context.Context, opts ImageModelCallOptions) (*ImageModelResult, error)
}

type VideoModel interface {
	Provider() string
	ModelID() string
	DoGenerateVideo(ctx context.Context, opts VideoModelCallOptions) (*VideoModelResult, error)
}

type SpeechModel interface {
	Provider() string
	ModelID() string
	DoGenerateSpeech(ctx context.Context, opts SpeechModelCallOptions) (*SpeechModelResult, error)
}

type TranscriptionModel interface {
	Provider() string
	ModelID() string
	DoTranscribe(ctx context.Context, opts TranscriptionModelCallOptions) (*TranscriptionModelResult, error)
}

type RerankingModel interface {
	Provider() string
	ModelID() string
	DoRerank(ctx context.Context, opts RerankingModelCallOptions) (*RerankingModelResult, error)
}

type EmbeddingModelCallOptions struct {
	Values          []string
	ProviderOptions ProviderOptions
	Headers         map[string]string
}

type EmbeddingModelResult struct {
	Embeddings       [][]float64
	Usage            EmbeddingUsage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

type GeneratedFile struct {
	Data      []byte
	URL       string
	MediaType string
	Filename  string
}

type ImageModelCallOptions struct {
	Prompt          string
	Images          []FilePart
	Size            string
	AspectRatio     string
	Seed            *int
	Headers         map[string]string
	ProviderOptions ProviderOptions
}

type ImageModelResult struct {
	Images           []GeneratedFile
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

type VideoModelCallOptions struct {
	Prompt          string
	Image           *FilePart
	Duration        string
	Size            string
	AspectRatio     string
	Seed            *int
	Headers         map[string]string
	ProviderOptions ProviderOptions
}

type VideoModelResult struct {
	Videos           []GeneratedFile
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

type SpeechModelCallOptions struct {
	Text            string
	Voice           string
	Speed           *float64
	Headers         map[string]string
	ProviderOptions ProviderOptions
}

type SpeechModelResult struct {
	Audio            GeneratedFile
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

type TranscriptionModelCallOptions struct {
	Audio           FilePart
	Language        string
	Prompt          string
	Headers         map[string]string
	ProviderOptions ProviderOptions
}

type TranscriptionModelResult struct {
	Text             string
	Segments         []TranscriptionSegment
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

type TranscriptionSegment struct {
	Text  string
	Start float64
	End   float64
}

type RerankingModelCallOptions struct {
	Query           string
	Documents       []string
	TopN            *int
	Headers         map[string]string
	ProviderOptions ProviderOptions
}

type RerankingModelResult struct {
	Results          []RerankingResult
	Usage            Usage
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

type RerankingResult struct {
	Index     int
	Document  string
	Score     float64
	Relevance any
}

type EmbeddingUsage struct {
	Tokens *int
}

type StreamPart struct {
	Type             string
	ID               string
	TextDelta        string
	PartialOutput    any
	Element          any
	ReasoningDelta   string
	ToolCallID       string
	ToolName         string
	ToolInputDelta   string
	ToolInput        string
	Content          Part
	FinishReason     FinishReason
	Usage            Usage
	Warnings         []Warning
	Request          RequestMetadata
	Response         ResponseMetadata
	ProviderMetadata ProviderMetadata
	Raw              any
	AbortReason      string
	Err              error
}

type ModelTool struct {
	Type            string
	Name            string
	Description     string
	InputSchema     any
	InputExamples   []any
	Strict          *bool
	ProviderOptions ProviderOptions
	ID              string
	Args            any
}

type ToolChoice struct {
	Type     string
	ToolName string
}

func AutoToolChoice() ToolChoice     { return ToolChoice{Type: "auto"} }
func RequiredToolChoice() ToolChoice { return ToolChoice{Type: "required"} }
func NoToolChoice() ToolChoice       { return ToolChoice{Type: "none"} }
func ToolChoiceFor(name string) ToolChoice {
	return ToolChoice{Type: "tool", ToolName: name}
}

var (
	ErrInvalidPrompt         = errors.New("invalid prompt")
	ErrInvalidArgument       = errors.New("invalid argument")
	ErrInvalidDataContent    = errors.New("invalid data content")
	ErrInvalidMessageRole    = errors.New("invalid message role")
	ErrMessageConversion     = errors.New("message conversion error")
	ErrMissingToolResults    = errors.New("missing tool results")
	ErrNoSuchTool            = errors.New("no such tool")
	ErrNoSuchProvider        = errors.New("no such provider")
	ErrNoSuchModel           = errors.New("no such model")
	ErrInvalidToolInput      = errors.New("invalid tool input")
	ErrInvalidResponseData   = errors.New("invalid response data")
	ErrAPICall               = errors.New("api call error")
	ErrRetry                 = errors.New("retry error")
	ErrDownload              = errors.New("download error")
	ErrGateway               = errors.New("gateway error")
	ErrGatewayAuthentication = errors.New("gateway authentication error")
	ErrUnsupportedFunction   = errors.New("unsupported functionality")
	ErrNoOutputGenerated     = errors.New("no output generated")
	ErrNoObjectGenerated     = errors.New("no object generated")
	ErrNoImageGenerated      = errors.New("no image generated")
	ErrNoVideoGenerated      = errors.New("no video generated")
	ErrNoSpeechGenerated     = errors.New("no speech generated")
	ErrNoTranscriptGenerated = errors.New("no transcript generated")
)

type SDKError struct {
	Kind    error
	Message string
	Cause   error
}

func (e *SDKError) Error() string {
	if e == nil {
		return ""
	}
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s: %v", e.Kind, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s: %s", e.Kind, e.Message)
}

func (e *SDKError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Kind
}

func intPtr(v int) *int           { return &v }
func floatPtr(v float64) *float64 { return &v }
func cloneJSONValue(v any) any {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		return v
	}
	return out
}
