package ai

import (
	"context"
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
)

type LanguageModelMiddleware interface {
	WrapGenerate(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next GenerateMiddlewareNext) (*LanguageModelGenerateResult, error)
	WrapStream(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next StreamMiddlewareNext) (*LanguageModelStreamResult, error)
}

type GenerateMiddlewareNext func(context.Context, LanguageModelCallOptions) (*LanguageModelGenerateResult, error)
type StreamMiddlewareNext func(context.Context, LanguageModelCallOptions) (*LanguageModelStreamResult, error)
type EmbedMiddlewareNext func(context.Context, EmbeddingModelCallOptions) (*EmbeddingModelResult, error)
type ImageGenerateMiddlewareNext func(context.Context, ImageModelCallOptions) (*ImageModelResult, error)
type VideoGenerateMiddlewareNext func(context.Context, VideoModelCallOptions) (*VideoModelResult, error)
type SpeechGenerateMiddlewareNext func(context.Context, SpeechModelCallOptions) (*SpeechModelResult, error)
type TranscriptionMiddlewareNext func(context.Context, TranscriptionModelCallOptions) (*TranscriptionModelResult, error)
type RerankMiddlewareNext func(context.Context, RerankingModelCallOptions) (*RerankingModelResult, error)

type EmbeddingModelMiddleware interface {
	WrapEmbed(ctx context.Context, model EmbeddingModel, opts EmbeddingModelCallOptions, next EmbedMiddlewareNext) (*EmbeddingModelResult, error)
	OverrideEmbeddingProvider(model EmbeddingModel) string
	OverrideEmbeddingModelID(model EmbeddingModel) string
	OverrideMaxEmbeddingsPerCall(model EmbeddingModel) int
}

type ImageModelMiddleware interface {
	WrapGenerateImage(ctx context.Context, model ImageModel, opts ImageModelCallOptions, next ImageGenerateMiddlewareNext) (*ImageModelResult, error)
	OverrideImageProvider(model ImageModel) string
	OverrideImageModelID(model ImageModel) string
}

type VideoModelMiddleware interface {
	WrapGenerateVideo(ctx context.Context, model VideoModel, opts VideoModelCallOptions, next VideoGenerateMiddlewareNext) (*VideoModelResult, error)
	OverrideVideoProvider(model VideoModel) string
	OverrideVideoModelID(model VideoModel) string
}

type SpeechModelMiddleware interface {
	WrapGenerateSpeech(ctx context.Context, model SpeechModel, opts SpeechModelCallOptions, next SpeechGenerateMiddlewareNext) (*SpeechModelResult, error)
	OverrideSpeechProvider(model SpeechModel) string
	OverrideSpeechModelID(model SpeechModel) string
}

type TranscriptionModelMiddleware interface {
	WrapTranscribe(ctx context.Context, model TranscriptionModel, opts TranscriptionModelCallOptions, next TranscriptionMiddlewareNext) (*TranscriptionModelResult, error)
	OverrideTranscriptionProvider(model TranscriptionModel) string
	OverrideTranscriptionModelID(model TranscriptionModel) string
}

type RerankingModelMiddleware interface {
	WrapRerank(ctx context.Context, model RerankingModel, opts RerankingModelCallOptions, next RerankMiddlewareNext) (*RerankingModelResult, error)
	OverrideRerankingProvider(model RerankingModel) string
	OverrideRerankingModelID(model RerankingModel) string
}

type MiddlewareFunc struct {
	Generate func(context.Context, LanguageModel, LanguageModelCallOptions, GenerateMiddlewareNext) (*LanguageModelGenerateResult, error)
	Stream   func(context.Context, LanguageModel, LanguageModelCallOptions, StreamMiddlewareNext) (*LanguageModelStreamResult, error)
}

func (m MiddlewareFunc) WrapGenerate(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next GenerateMiddlewareNext) (*LanguageModelGenerateResult, error) {
	if m.Generate == nil {
		return next(ctx, opts)
	}
	return m.Generate(ctx, model, opts, next)
}

func (m MiddlewareFunc) WrapStream(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next StreamMiddlewareNext) (*LanguageModelStreamResult, error) {
	if m.Stream == nil {
		return next(ctx, opts)
	}
	return m.Stream(ctx, model, opts, next)
}

func WrapLanguageModel(model LanguageModel, middleware ...LanguageModelMiddleware) LanguageModel {
	if len(middleware) == 0 {
		return model
	}
	return wrappedLanguageModel{model: model, middleware: middleware}
}

type EmbeddingMiddlewareFunc struct {
	Embed                        func(context.Context, EmbeddingModel, EmbeddingModelCallOptions, EmbedMiddlewareNext) (*EmbeddingModelResult, error)
	Provider                     func(EmbeddingModel) string
	ModelID                      func(EmbeddingModel) string
	MaxEmbeddingsPerCallOverride func(EmbeddingModel) int
}

func (m EmbeddingMiddlewareFunc) WrapEmbed(ctx context.Context, model EmbeddingModel, opts EmbeddingModelCallOptions, next EmbedMiddlewareNext) (*EmbeddingModelResult, error) {
	if m.Embed == nil {
		return next(ctx, opts)
	}
	return m.Embed(ctx, model, opts, next)
}

func (m EmbeddingMiddlewareFunc) OverrideEmbeddingProvider(model EmbeddingModel) string {
	if m.Provider == nil {
		return ""
	}
	return m.Provider(model)
}

func (m EmbeddingMiddlewareFunc) OverrideEmbeddingModelID(model EmbeddingModel) string {
	if m.ModelID == nil {
		return ""
	}
	return m.ModelID(model)
}

func (m EmbeddingMiddlewareFunc) OverrideMaxEmbeddingsPerCall(model EmbeddingModel) int {
	if m.MaxEmbeddingsPerCallOverride == nil {
		return 0
	}
	return m.MaxEmbeddingsPerCallOverride(model)
}

func WrapEmbeddingModel(model EmbeddingModel, middleware ...EmbeddingModelMiddleware) EmbeddingModel {
	if len(middleware) == 0 {
		return model
	}
	return wrappedEmbeddingModel{model: model, middleware: middleware}
}

type wrappedEmbeddingModel struct {
	model      EmbeddingModel
	middleware []EmbeddingModelMiddleware
}

func (m wrappedEmbeddingModel) Provider() string {
	for _, middleware := range m.middleware {
		if provider := middleware.OverrideEmbeddingProvider(m.model); provider != "" {
			return provider
		}
	}
	return m.model.Provider()
}

func (m wrappedEmbeddingModel) ModelID() string {
	for _, middleware := range m.middleware {
		if modelID := middleware.OverrideEmbeddingModelID(m.model); modelID != "" {
			return modelID
		}
	}
	return m.model.ModelID()
}

func (m wrappedEmbeddingModel) MaxEmbeddingsPerCall() int {
	for _, middleware := range m.middleware {
		if max := middleware.OverrideMaxEmbeddingsPerCall(m.model); max > 0 {
			return max
		}
	}
	return m.model.MaxEmbeddingsPerCall()
}

func (m wrappedEmbeddingModel) DoEmbed(ctx context.Context, opts EmbeddingModelCallOptions) (*EmbeddingModelResult, error) {
	var next EmbedMiddlewareNext
	index := 0
	next = func(ctx context.Context, opts EmbeddingModelCallOptions) (*EmbeddingModelResult, error) {
		if index >= len(m.middleware) {
			return m.model.DoEmbed(ctx, opts)
		}
		current := m.middleware[index]
		index++
		return current.WrapEmbed(ctx, m.model, opts, next)
	}
	return next(ctx, opts)
}

type ImageMiddlewareFunc struct {
	Generate func(context.Context, ImageModel, ImageModelCallOptions, ImageGenerateMiddlewareNext) (*ImageModelResult, error)
	Provider func(ImageModel) string
	ModelID  func(ImageModel) string
}

func (m ImageMiddlewareFunc) WrapGenerateImage(ctx context.Context, model ImageModel, opts ImageModelCallOptions, next ImageGenerateMiddlewareNext) (*ImageModelResult, error) {
	if m.Generate == nil {
		return next(ctx, opts)
	}
	return m.Generate(ctx, model, opts, next)
}

func (m ImageMiddlewareFunc) OverrideImageProvider(model ImageModel) string {
	if m.Provider == nil {
		return ""
	}
	return m.Provider(model)
}

func (m ImageMiddlewareFunc) OverrideImageModelID(model ImageModel) string {
	if m.ModelID == nil {
		return ""
	}
	return m.ModelID(model)
}

func WrapImageModel(model ImageModel, middleware ...ImageModelMiddleware) ImageModel {
	if len(middleware) == 0 {
		return model
	}
	return wrappedImageModel{model: model, middleware: middleware}
}

type VideoMiddlewareFunc struct {
	Generate func(context.Context, VideoModel, VideoModelCallOptions, VideoGenerateMiddlewareNext) (*VideoModelResult, error)
	Provider func(VideoModel) string
	ModelID  func(VideoModel) string
}

func (m VideoMiddlewareFunc) WrapGenerateVideo(ctx context.Context, model VideoModel, opts VideoModelCallOptions, next VideoGenerateMiddlewareNext) (*VideoModelResult, error) {
	if m.Generate == nil {
		return next(ctx, opts)
	}
	return m.Generate(ctx, model, opts, next)
}

func (m VideoMiddlewareFunc) OverrideVideoProvider(model VideoModel) string {
	if m.Provider == nil {
		return ""
	}
	return m.Provider(model)
}

func (m VideoMiddlewareFunc) OverrideVideoModelID(model VideoModel) string {
	if m.ModelID == nil {
		return ""
	}
	return m.ModelID(model)
}

func WrapVideoModel(model VideoModel, middleware ...VideoModelMiddleware) VideoModel {
	if len(middleware) == 0 {
		return model
	}
	return wrappedVideoModel{model: model, middleware: middleware}
}

type SpeechMiddlewareFunc struct {
	Generate func(context.Context, SpeechModel, SpeechModelCallOptions, SpeechGenerateMiddlewareNext) (*SpeechModelResult, error)
	Provider func(SpeechModel) string
	ModelID  func(SpeechModel) string
}

func (m SpeechMiddlewareFunc) WrapGenerateSpeech(ctx context.Context, model SpeechModel, opts SpeechModelCallOptions, next SpeechGenerateMiddlewareNext) (*SpeechModelResult, error) {
	if m.Generate == nil {
		return next(ctx, opts)
	}
	return m.Generate(ctx, model, opts, next)
}

func (m SpeechMiddlewareFunc) OverrideSpeechProvider(model SpeechModel) string {
	if m.Provider == nil {
		return ""
	}
	return m.Provider(model)
}

func (m SpeechMiddlewareFunc) OverrideSpeechModelID(model SpeechModel) string {
	if m.ModelID == nil {
		return ""
	}
	return m.ModelID(model)
}

func WrapSpeechModel(model SpeechModel, middleware ...SpeechModelMiddleware) SpeechModel {
	if len(middleware) == 0 {
		return model
	}
	return wrappedSpeechModel{model: model, middleware: middleware}
}

type TranscriptionMiddlewareFunc struct {
	Transcribe func(context.Context, TranscriptionModel, TranscriptionModelCallOptions, TranscriptionMiddlewareNext) (*TranscriptionModelResult, error)
	Provider   func(TranscriptionModel) string
	ModelID    func(TranscriptionModel) string
}

func (m TranscriptionMiddlewareFunc) WrapTranscribe(ctx context.Context, model TranscriptionModel, opts TranscriptionModelCallOptions, next TranscriptionMiddlewareNext) (*TranscriptionModelResult, error) {
	if m.Transcribe == nil {
		return next(ctx, opts)
	}
	return m.Transcribe(ctx, model, opts, next)
}

func (m TranscriptionMiddlewareFunc) OverrideTranscriptionProvider(model TranscriptionModel) string {
	if m.Provider == nil {
		return ""
	}
	return m.Provider(model)
}

func (m TranscriptionMiddlewareFunc) OverrideTranscriptionModelID(model TranscriptionModel) string {
	if m.ModelID == nil {
		return ""
	}
	return m.ModelID(model)
}

func WrapTranscriptionModel(model TranscriptionModel, middleware ...TranscriptionModelMiddleware) TranscriptionModel {
	if len(middleware) == 0 {
		return model
	}
	return wrappedTranscriptionModel{model: model, middleware: middleware}
}

type RerankingMiddlewareFunc struct {
	Rerank   func(context.Context, RerankingModel, RerankingModelCallOptions, RerankMiddlewareNext) (*RerankingModelResult, error)
	Provider func(RerankingModel) string
	ModelID  func(RerankingModel) string
}

func (m RerankingMiddlewareFunc) WrapRerank(ctx context.Context, model RerankingModel, opts RerankingModelCallOptions, next RerankMiddlewareNext) (*RerankingModelResult, error) {
	if m.Rerank == nil {
		return next(ctx, opts)
	}
	return m.Rerank(ctx, model, opts, next)
}

func (m RerankingMiddlewareFunc) OverrideRerankingProvider(model RerankingModel) string {
	if m.Provider == nil {
		return ""
	}
	return m.Provider(model)
}

func (m RerankingMiddlewareFunc) OverrideRerankingModelID(model RerankingModel) string {
	if m.ModelID == nil {
		return ""
	}
	return m.ModelID(model)
}

func WrapRerankingModel(model RerankingModel, middleware ...RerankingModelMiddleware) RerankingModel {
	if len(middleware) == 0 {
		return model
	}
	return wrappedRerankingModel{model: model, middleware: middleware}
}

type wrappedImageModel struct {
	model      ImageModel
	middleware []ImageModelMiddleware
}

func (m wrappedImageModel) Provider() string {
	for _, middleware := range m.middleware {
		if provider := middleware.OverrideImageProvider(m.model); provider != "" {
			return provider
		}
	}
	return m.model.Provider()
}

func (m wrappedImageModel) ModelID() string {
	for _, middleware := range m.middleware {
		if modelID := middleware.OverrideImageModelID(m.model); modelID != "" {
			return modelID
		}
	}
	return m.model.ModelID()
}

func (m wrappedImageModel) DoGenerateImage(ctx context.Context, opts ImageModelCallOptions) (*ImageModelResult, error) {
	var next ImageGenerateMiddlewareNext
	index := 0
	next = func(ctx context.Context, opts ImageModelCallOptions) (*ImageModelResult, error) {
		if index >= len(m.middleware) {
			return m.model.DoGenerateImage(ctx, opts)
		}
		current := m.middleware[index]
		index++
		return current.WrapGenerateImage(ctx, m.model, opts, next)
	}
	return next(ctx, opts)
}

type wrappedLanguageModel struct {
	model      LanguageModel
	middleware []LanguageModelMiddleware
}

func (m wrappedLanguageModel) Provider() string { return m.model.Provider() }
func (m wrappedLanguageModel) ModelID() string  { return m.model.ModelID() }
func (m wrappedLanguageModel) SupportedURLs(ctx context.Context) (map[string][]string, error) {
	return m.model.SupportedURLs(ctx)
}

func (m wrappedLanguageModel) DoGenerate(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
	var next GenerateMiddlewareNext
	index := 0
	next = func(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		if index >= len(m.middleware) {
			return m.model.DoGenerate(ctx, opts)
		}
		current := m.middleware[index]
		index++
		return current.WrapGenerate(ctx, m.model, opts, next)
	}
	return next(ctx, opts)
}

func (m wrappedLanguageModel) DoStream(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
	var next StreamMiddlewareNext
	index := 0
	next = func(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		if index >= len(m.middleware) {
			return m.model.DoStream(ctx, opts)
		}
		current := m.middleware[index]
		index++
		return current.WrapStream(ctx, m.model, opts, next)
	}
	return next(ctx, opts)
}

func DefaultSettings(defaults LanguageModelCallOptions) LanguageModelMiddleware {
	return MiddlewareFunc{
		Generate: func(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next GenerateMiddlewareNext) (*LanguageModelGenerateResult, error) {
			return next(ctx, mergeCallSettings(defaults, opts))
		},
		Stream: func(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next StreamMiddlewareNext) (*LanguageModelStreamResult, error) {
			return next(ctx, mergeCallSettings(defaults, opts))
		},
	}
}

type ExtractJSONMiddlewareOptions struct {
	Transform func(string) string
}

func ExtractJSONMiddleware(options ...ExtractJSONMiddlewareOptions) LanguageModelMiddleware {
	var opts ExtractJSONMiddlewareOptions
	if len(options) > 0 {
		opts = options[0]
	}
	transform := opts.Transform
	if transform == nil {
		transform = defaultExtractJSONTransform
	}
	return MiddlewareFunc{
		Generate: func(ctx context.Context, model LanguageModel, callOpts LanguageModelCallOptions, next GenerateMiddlewareNext) (*LanguageModelGenerateResult, error) {
			result, err := next(ctx, callOpts)
			if err != nil || result == nil {
				return result, err
			}
			result.Content = transformTextParts(result.Content, transform)
			return result, nil
		},
		Stream: func(ctx context.Context, model LanguageModel, callOpts LanguageModelCallOptions, next StreamMiddlewareNext) (*LanguageModelStreamResult, error) {
			return next(ctx, callOpts)
		},
	}
}

func defaultExtractJSONTransform(text string) string {
	text = strings.TrimSpace(text)
	text = regexp.MustCompile("^```(?:json)?\\s*\\n?").ReplaceAllString(text, "")
	text = regexp.MustCompile("\\n?```\\s*$").ReplaceAllString(text, "")
	return strings.TrimSpace(text)
}

type ExtractReasoningMiddlewareOptions struct {
	TagName            string
	Separator          string
	StartWithReasoning bool
}

func ExtractReasoningMiddleware(options ExtractReasoningMiddlewareOptions) LanguageModelMiddleware {
	separator := options.Separator
	if separator == "" {
		separator = "\n"
	}
	openingTag := "<" + options.TagName + ">"
	closingTag := "</" + options.TagName + ">"
	return MiddlewareFunc{
		Generate: func(ctx context.Context, model LanguageModel, callOpts LanguageModelCallOptions, next GenerateMiddlewareNext) (*LanguageModelGenerateResult, error) {
			if options.TagName == "" {
				return nil, &SDKError{Kind: ErrInvalidArgument, Message: "reasoning tag name is required"}
			}
			result, err := next(ctx, callOpts)
			if err != nil || result == nil {
				return result, err
			}
			result.Content = extractReasoningParts(result.Content, openingTag, closingTag, separator, options.StartWithReasoning)
			return result, nil
		},
		Stream: func(ctx context.Context, model LanguageModel, callOpts LanguageModelCallOptions, next StreamMiddlewareNext) (*LanguageModelStreamResult, error) {
			if options.TagName == "" {
				return nil, &SDKError{Kind: ErrInvalidArgument, Message: "reasoning tag name is required"}
			}
			return next(ctx, callOpts)
		},
	}
}

func SimulateStreamingMiddleware() LanguageModelMiddleware {
	return MiddlewareFunc{
		Stream: func(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next StreamMiddlewareNext) (*LanguageModelStreamResult, error) {
			result, err := model.DoGenerate(ctx, opts)
			if err != nil {
				return nil, err
			}
			out := make(chan StreamPart)
			go func() {
				defer close(out)
				out <- StreamPart{Type: "stream-start", Warnings: result.Warnings}
				out <- StreamPart{Type: "response-metadata", Response: result.Response}
				id := 0
				for _, part := range result.Content {
					switch p := part.(type) {
					case TextPart:
						if p.Text != "" {
							partID := stringID(id)
							out <- StreamPart{Type: "text-start", ID: partID}
							out <- StreamPart{Type: "text-delta", ID: partID, TextDelta: p.Text}
							out <- StreamPart{Type: "text-end", ID: partID}
							id++
						}
					case ReasoningPart:
						partID := stringID(id)
						out <- StreamPart{Type: "reasoning-start", ID: partID, ProviderMetadata: p.ProviderMetadata}
						out <- StreamPart{Type: "reasoning-delta", ID: partID, ReasoningDelta: p.Text}
						out <- StreamPart{Type: "reasoning-end", ID: partID}
						id++
					case ToolCallPart:
						out <- StreamPart{Type: "tool-call", ToolCallID: p.ToolCallID, ToolName: p.ToolName, ToolInput: p.InputJSON(), ProviderMetadata: p.ProviderMetadata}
					default:
						out <- StreamPart{Type: part.PartType(), Content: part}
					}
				}
				out <- StreamPart{Type: "finish", FinishReason: result.FinishReason, Usage: result.Usage, ProviderMetadata: result.ProviderMetadata}
			}()
			return &LanguageModelStreamResult{Stream: out, Request: result.Request, Response: result.Response}, nil
		},
	}
}

type AddToolInputExamplesMiddlewareOptions struct {
	Prefix              string
	Format              func(example any, index int) (string, error)
	RemoveInputExamples *bool
}

func AddToolInputExamplesMiddleware(options ...AddToolInputExamplesMiddlewareOptions) LanguageModelMiddleware {
	var opts AddToolInputExamplesMiddlewareOptions
	if len(options) > 0 {
		opts = options[0]
	}
	prefix := opts.Prefix
	if prefix == "" {
		prefix = "Input Examples:"
	}
	format := opts.Format
	if format == nil {
		format = defaultFormatToolInputExample
	}
	remove := true
	if opts.RemoveInputExamples != nil {
		remove = *opts.RemoveInputExamples
	}
	transform := func(callOpts LanguageModelCallOptions) (LanguageModelCallOptions, error) {
		if len(callOpts.Tools) == 0 {
			return callOpts, nil
		}
		tools := make([]ModelTool, len(callOpts.Tools))
		copy(tools, callOpts.Tools)
		for i, tool := range tools {
			if tool.Type != "function" || len(tool.InputExamples) == 0 {
				continue
			}
			formatted := make([]string, 0, len(tool.InputExamples))
			for j, example := range tool.InputExamples {
				text, err := format(example, j)
				if err != nil {
					return callOpts, err
				}
				formatted = append(formatted, text)
			}
			examplesSection := prefix + "\n" + strings.Join(formatted, "\n")
			if tool.Description != "" {
				tool.Description += "\n\n" + examplesSection
			} else {
				tool.Description = examplesSection
			}
			if remove {
				tool.InputExamples = nil
			}
			tools[i] = tool
		}
		callOpts.Tools = tools
		return callOpts, nil
	}
	return MiddlewareFunc{
		Generate: func(ctx context.Context, model LanguageModel, callOpts LanguageModelCallOptions, next GenerateMiddlewareNext) (*LanguageModelGenerateResult, error) {
			transformed, err := transform(callOpts)
			if err != nil {
				return nil, err
			}
			return next(ctx, transformed)
		},
		Stream: func(ctx context.Context, model LanguageModel, callOpts LanguageModelCallOptions, next StreamMiddlewareNext) (*LanguageModelStreamResult, error) {
			transformed, err := transform(callOpts)
			if err != nil {
				return nil, err
			}
			return next(ctx, transformed)
		},
	}
}

func transformTextParts(parts []Part, transform func(string) string) []Part {
	out := make([]Part, len(parts))
	for i, part := range parts {
		if text, ok := part.(TextPart); ok {
			text.Text = transform(text.Text)
			out[i] = text
			continue
		}
		out[i] = part
	}
	return out
}

func extractReasoningParts(parts []Part, openingTag, closingTag, separator string, startWithReasoning bool) []Part {
	pattern := regexp.MustCompile("(?s)" + regexp.QuoteMeta(openingTag) + "(.*?)" + regexp.QuoteMeta(closingTag))
	var out []Part
	for _, part := range parts {
		textPart, ok := part.(TextPart)
		if !ok {
			out = append(out, part)
			continue
		}
		text := textPart.Text
		if startWithReasoning {
			text = openingTag + text
		}
		matches := pattern.FindAllStringSubmatchIndex(text, -1)
		if len(matches) == 0 {
			out = append(out, part)
			continue
		}
		reasoning := make([]string, 0, len(matches))
		for _, match := range matches {
			reasoning = append(reasoning, text[match[2]:match[3]])
		}
		textWithoutReasoning := text
		for i := len(matches) - 1; i >= 0; i-- {
			match := matches[i]
			before := textWithoutReasoning[:match[0]]
			after := textWithoutReasoning[match[1]:]
			joiner := ""
			if before != "" && after != "" {
				joiner = separator
			}
			textWithoutReasoning = before + joiner + after
		}
		out = append(out, ReasoningPart{Text: strings.Join(reasoning, separator)})
		textPart.Text = textWithoutReasoning
		out = append(out, textPart)
	}
	return out
}

type reasoningStreamState struct {
	isFirstReasoning bool
	isFirstText      bool
	afterSwitch      bool
	isReasoning      bool
	buffer           string
	idCounter        int
	separator        string
	openingTag       string
	closingTag       string
}

func (s *reasoningStreamState) publishAvailable(out chan<- StreamPart, template StreamPart) {
	for {
		nextTag := s.openingTag
		if s.isReasoning {
			nextTag = s.closingTag
		}
		startIndex := potentialStartIndex(s.buffer, nextTag)
		if startIndex < 0 {
			s.publish(out, template, s.buffer)
			s.buffer = ""
			return
		}
		s.publish(out, template, s.buffer[:startIndex])
		if startIndex+len(nextTag) > len(s.buffer) {
			s.buffer = s.buffer[startIndex:]
			return
		}
		s.buffer = s.buffer[startIndex+len(nextTag):]
		s.isReasoning = !s.isReasoning
		s.afterSwitch = true
	}
}

func (s *reasoningStreamState) flush(out chan<- StreamPart, template StreamPart) {
	if s.buffer == "" {
		return
	}
	s.publish(out, template, s.buffer)
	s.buffer = ""
}

func (s *reasoningStreamState) publish(out chan<- StreamPart, template StreamPart, text string) {
	if text == "" {
		return
	}
	prefix := ""
	if s.afterSwitch {
		if s.isReasoning && !s.isFirstReasoning {
			prefix = s.separator
		}
		if !s.isReasoning && !s.isFirstText {
			prefix = s.separator
		}
	}
	text = prefix + text
	if s.isReasoning {
		out <- StreamPart{Type: "reasoning-delta", ID: stringID(s.idCounter), ReasoningDelta: text, ProviderMetadata: template.ProviderMetadata, Raw: template.Raw}
		s.isFirstReasoning = false
	} else {
		template.Type = "text-delta"
		template.TextDelta = text
		template.ReasoningDelta = ""
		out <- template
		s.isFirstText = false
	}
	s.afterSwitch = false
}

func potentialStartIndex(text, match string) int {
	if index := strings.Index(text, match); index >= 0 {
		return index
	}
	max := len(match) - 1
	if len(text) < max {
		max = len(text)
	}
	for length := max; length > 0; length-- {
		if strings.HasSuffix(text, match[:length]) {
			return len(text) - length
		}
	}
	return -1
}

func defaultFormatToolInputExample(example any, index int) (string, error) {
	if object, ok := example.(map[string]any); ok {
		if input, ok := object["input"]; ok {
			example = input
		}
	}
	b, err := json.Marshal(example)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func stringID(id int) string {
	return strconv.Itoa(id)
}

func mergeCallSettings(defaults, opts LanguageModelCallOptions) LanguageModelCallOptions {
	out := opts
	if out.MaxOutputTokens == nil {
		out.MaxOutputTokens = defaults.MaxOutputTokens
	}
	if out.Temperature == nil {
		out.Temperature = defaults.Temperature
	}
	if out.TopP == nil {
		out.TopP = defaults.TopP
	}
	if out.TopK == nil {
		out.TopK = defaults.TopK
	}
	if out.PresencePenalty == nil {
		out.PresencePenalty = defaults.PresencePenalty
	}
	if out.FrequencyPenalty == nil {
		out.FrequencyPenalty = defaults.FrequencyPenalty
	}
	if len(out.StopSequences) == 0 {
		out.StopSequences = defaults.StopSequences
	}
	if out.Seed == nil {
		out.Seed = defaults.Seed
	}
	if out.Reasoning == "" {
		out.Reasoning = defaults.Reasoning
	}
	if out.ResponseFormat == nil {
		out.ResponseFormat = defaults.ResponseFormat
	}
	if len(out.Headers) == 0 {
		out.Headers = defaults.Headers
	} else if len(defaults.Headers) > 0 {
		headers := map[string]string{}
		for key, value := range defaults.Headers {
			headers[key] = value
		}
		for key, value := range out.Headers {
			headers[key] = value
		}
		out.Headers = headers
	}
	out.ProviderOptions = mergeProviderOptions(cloneProviderOptions(defaults.ProviderOptions), out.ProviderOptions)
	return out
}

func DefaultEmbeddingSettings(defaults EmbeddingModelCallOptions) EmbeddingModelMiddleware {
	return EmbeddingMiddlewareFunc{
		Embed: func(ctx context.Context, model EmbeddingModel, opts EmbeddingModelCallOptions, next EmbedMiddlewareNext) (*EmbeddingModelResult, error) {
			return next(ctx, mergeEmbeddingSettings(defaults, opts))
		},
	}
}

func mergeEmbeddingSettings(defaults, opts EmbeddingModelCallOptions) EmbeddingModelCallOptions {
	out := opts
	if len(out.Headers) == 0 {
		out.Headers = defaults.Headers
	} else if len(defaults.Headers) > 0 {
		headers := map[string]string{}
		for key, value := range defaults.Headers {
			headers[key] = value
		}
		for key, value := range out.Headers {
			headers[key] = value
		}
		out.Headers = headers
	}
	out.ProviderOptions = mergeProviderOptions(cloneProviderOptions(defaults.ProviderOptions), out.ProviderOptions)
	return out
}

type WrappedProvider struct {
	Provider                     Provider
	LanguageModelMiddleware      []LanguageModelMiddleware
	EmbeddingModelMiddleware     []EmbeddingModelMiddleware
	ImageModelMiddleware         []ImageModelMiddleware
	VideoModelMiddleware         []VideoModelMiddleware
	SpeechModelMiddleware        []SpeechModelMiddleware
	TranscriptionModelMiddleware []TranscriptionModelMiddleware
	RerankingModelMiddleware     []RerankingModelMiddleware
}

func WrapProvider(provider Provider, languageModelMiddleware []LanguageModelMiddleware, imageModelMiddleware []ImageModelMiddleware) *WrappedProvider {
	return &WrappedProvider{
		Provider:                provider,
		LanguageModelMiddleware: languageModelMiddleware,
		ImageModelMiddleware:    imageModelMiddleware,
	}
}

func WrapProviderWithEmbedding(provider Provider, languageModelMiddleware []LanguageModelMiddleware, embeddingModelMiddleware []EmbeddingModelMiddleware, imageModelMiddleware []ImageModelMiddleware) *WrappedProvider {
	return &WrappedProvider{
		Provider:                 provider,
		LanguageModelMiddleware:  languageModelMiddleware,
		EmbeddingModelMiddleware: embeddingModelMiddleware,
		ImageModelMiddleware:     imageModelMiddleware,
	}
}

func WrapProviderWithMedia(provider Provider, languageModelMiddleware []LanguageModelMiddleware, embeddingModelMiddleware []EmbeddingModelMiddleware, imageModelMiddleware []ImageModelMiddleware, videoModelMiddleware []VideoModelMiddleware, speechModelMiddleware []SpeechModelMiddleware, transcriptionModelMiddleware []TranscriptionModelMiddleware, rerankingModelMiddleware []RerankingModelMiddleware) *WrappedProvider {
	return &WrappedProvider{
		Provider:                     provider,
		LanguageModelMiddleware:      languageModelMiddleware,
		EmbeddingModelMiddleware:     embeddingModelMiddleware,
		ImageModelMiddleware:         imageModelMiddleware,
		VideoModelMiddleware:         videoModelMiddleware,
		SpeechModelMiddleware:        speechModelMiddleware,
		TranscriptionModelMiddleware: transcriptionModelMiddleware,
		RerankingModelMiddleware:     rerankingModelMiddleware,
	}
}

func (p *WrappedProvider) LanguageModel(modelID string) LanguageModel {
	model := p.Provider.LanguageModel(modelID)
	if model == nil {
		return nil
	}
	return WrapLanguageModel(model, p.LanguageModelMiddleware...)
}

func (p *WrappedProvider) EmbeddingModel(modelID string) EmbeddingModel {
	provider, ok := p.Provider.(EmbeddingProvider)
	if !ok {
		return nil
	}
	model := provider.EmbeddingModel(modelID)
	if model == nil {
		return nil
	}
	return WrapEmbeddingModel(model, p.EmbeddingModelMiddleware...)
}

func (p *WrappedProvider) ImageModel(modelID string) ImageModel {
	provider, ok := p.Provider.(ImageProvider)
	if !ok {
		return nil
	}
	model := provider.ImageModel(modelID)
	if model == nil {
		return nil
	}
	return WrapImageModel(model, p.ImageModelMiddleware...)
}

func (p *WrappedProvider) VideoModel(modelID string) VideoModel {
	provider, ok := p.Provider.(VideoProvider)
	if !ok {
		return nil
	}
	model := provider.VideoModel(modelID)
	if model == nil {
		return nil
	}
	return WrapVideoModel(model, p.VideoModelMiddleware...)
}

func (p *WrappedProvider) SpeechModel(modelID string) SpeechModel {
	provider, ok := p.Provider.(SpeechProvider)
	if !ok {
		return nil
	}
	model := provider.SpeechModel(modelID)
	if model == nil {
		return nil
	}
	return WrapSpeechModel(model, p.SpeechModelMiddleware...)
}

func (p *WrappedProvider) TranscriptionModel(modelID string) TranscriptionModel {
	provider, ok := p.Provider.(TranscriptionProvider)
	if !ok {
		return nil
	}
	model := provider.TranscriptionModel(modelID)
	if model == nil {
		return nil
	}
	return WrapTranscriptionModel(model, p.TranscriptionModelMiddleware...)
}

func (p *WrappedProvider) RerankingModel(modelID string) RerankingModel {
	provider, ok := p.Provider.(RerankingProvider)
	if !ok {
		return nil
	}
	model := provider.RerankingModel(modelID)
	if model == nil {
		return nil
	}
	return WrapRerankingModel(model, p.RerankingModelMiddleware...)
}
