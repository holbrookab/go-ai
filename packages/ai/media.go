package ai

import "context"

type GenerateImageOptions struct {
	Model            ImageModel
	Prompt           string
	Images           []FilePart
	Size             string
	AspectRatio      string
	Seed             *int
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type GenerateImageResult struct {
	Images           []GeneratedFile
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

func GenerateImage(ctx context.Context, opts GenerateImageOptions) (imageResult *GenerateImageResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventGenerateImageStart, OperationGenerateImage, opts.Model, map[string]any{
		"prompt":       opts.Prompt,
		"image_count":  len(opts.Images),
		"size":         opts.Size,
		"aspect_ratio": opts.AspectRatio,
	})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventGenerateImageError, OperationGenerateImage, err)
		}
	}()
	if opts.Model == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "model is required"}
	}
	result, err := opts.Model.DoGenerateImage(ctx, ImageModelCallOptions{
		Prompt:          opts.Prompt,
		Images:          opts.Images,
		Size:            opts.Size,
		AspectRatio:     opts.AspectRatio,
		Seed:            opts.Seed,
		Headers:         withUserAgent(opts.Headers, "go-ai/"+Version),
		ProviderOptions: opts.ProviderOptions,
	})
	if err != nil {
		return nil, err
	}
	if result == nil || len(result.Images) == 0 {
		var responses []ResponseMetadata
		if result != nil {
			responses = append(responses, result.Response)
		}
		return nil, NewNoImageGeneratedError(NoImageGeneratedErrorOptions{Message: "model returned no images", Responses: responses})
	}
	LogWarnings(result.Warnings, opts.Model.Provider(), opts.Model.ModelID())
	imageResult = &GenerateImageResult{
		Images:           result.Images,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Response,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventGenerateImageFinish, OperationGenerateImage, imageResult, map[string]any{
		"image_count": len(imageResult.Images),
	})
	return imageResult, nil
}

type GenerateVideoOptions struct {
	Model            VideoModel
	Prompt           string
	Image            *FilePart
	Duration         string
	Size             string
	AspectRatio      string
	Seed             *int
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type GenerateVideoResult struct {
	Videos           []GeneratedFile
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

func GenerateVideo(ctx context.Context, opts GenerateVideoOptions) (videoResult *GenerateVideoResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventGenerateVideoStart, OperationGenerateVideo, opts.Model, map[string]any{
		"prompt":       opts.Prompt,
		"has_image":    opts.Image != nil,
		"duration":     opts.Duration,
		"size":         opts.Size,
		"aspect_ratio": opts.AspectRatio,
	})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventGenerateVideoError, OperationGenerateVideo, err)
		}
	}()
	if opts.Model == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "model is required"}
	}
	result, err := opts.Model.DoGenerateVideo(ctx, VideoModelCallOptions{
		Prompt:          opts.Prompt,
		Image:           opts.Image,
		Duration:        opts.Duration,
		Size:            opts.Size,
		AspectRatio:     opts.AspectRatio,
		Seed:            opts.Seed,
		Headers:         withUserAgent(opts.Headers, "go-ai/"+Version),
		ProviderOptions: opts.ProviderOptions,
	})
	if err != nil {
		return nil, err
	}
	if result == nil || len(result.Videos) == 0 {
		var responses []ResponseMetadata
		if result != nil {
			responses = append(responses, result.Response)
		}
		return nil, NewNoVideoGeneratedError(NoVideoGeneratedErrorOptions{Message: "model returned no videos", Responses: responses})
	}
	LogWarnings(result.Warnings, opts.Model.Provider(), opts.Model.ModelID())
	videoResult = &GenerateVideoResult{
		Videos:           result.Videos,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Response,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventGenerateVideoFinish, OperationGenerateVideo, videoResult, map[string]any{
		"video_count": len(videoResult.Videos),
	})
	return videoResult, nil
}

type GenerateSpeechOptions struct {
	Model            SpeechModel
	Text             string
	Voice            string
	Speed            *float64
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type GenerateSpeechResult struct {
	Audio            GeneratedFile
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

func GenerateSpeech(ctx context.Context, opts GenerateSpeechOptions) (speechResult *GenerateSpeechResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventGenerateSpeechStart, OperationGenerateSpeech, opts.Model, map[string]any{
		"text":  opts.Text,
		"voice": opts.Voice,
	})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventGenerateSpeechError, OperationGenerateSpeech, err)
		}
	}()
	if opts.Model == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "model is required"}
	}
	result, err := opts.Model.DoGenerateSpeech(ctx, SpeechModelCallOptions{
		Text:            opts.Text,
		Voice:           opts.Voice,
		Speed:           opts.Speed,
		Headers:         withUserAgent(opts.Headers, "go-ai/"+Version),
		ProviderOptions: opts.ProviderOptions,
	})
	if err != nil {
		return nil, err
	}
	if result == nil || (len(result.Audio.Data) == 0 && result.Audio.URL == "") {
		var responses []ResponseMetadata
		if result != nil {
			responses = append(responses, result.Response)
		}
		return nil, NewNoSpeechGeneratedError(responses)
	}
	LogWarnings(result.Warnings, opts.Model.Provider(), opts.Model.ModelID())
	speechResult = &GenerateSpeechResult{
		Audio:            result.Audio,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Response,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventGenerateSpeechFinish, OperationGenerateSpeech, speechResult, map[string]any{
		"media_type": speechResult.Audio.MediaType,
	})
	return speechResult, nil
}

type TranscribeOptions struct {
	Model            TranscriptionModel
	Audio            FilePart
	Language         string
	Prompt           string
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type TranscribeResult struct {
	Text             string
	Segments         []TranscriptionSegment
	Warnings         []Warning
	ProviderMetadata ProviderMetadata
	Response         ResponseMetadata
}

func Transcribe(ctx context.Context, opts TranscribeOptions) (transcribeResult *TranscribeResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventTranscribeStart, OperationTranscribe, opts.Model, map[string]any{
		"language": opts.Language,
		"prompt":   opts.Prompt,
	})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventTranscribeError, OperationTranscribe, err)
		}
	}()
	if opts.Model == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "model is required"}
	}
	result, err := opts.Model.DoTranscribe(ctx, TranscriptionModelCallOptions{
		Audio:           opts.Audio,
		Language:        opts.Language,
		Prompt:          opts.Prompt,
		Headers:         withUserAgent(opts.Headers, "go-ai/"+Version),
		ProviderOptions: opts.ProviderOptions,
	})
	if err != nil {
		return nil, err
	}
	if result == nil || result.Text == "" {
		var responses []ResponseMetadata
		if result != nil {
			responses = append(responses, result.Response)
		}
		return nil, NewNoTranscriptGeneratedError(responses)
	}
	LogWarnings(result.Warnings, opts.Model.Provider(), opts.Model.ModelID())
	transcribeResult = &TranscribeResult{
		Text:             result.Text,
		Segments:         result.Segments,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         result.Response,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventTranscribeFinish, OperationTranscribe, transcribeResult, map[string]any{
		"segment_count": len(transcribeResult.Segments),
	})
	return transcribeResult, nil
}

type RerankOptions struct {
	Model            RerankingModel
	Query            string
	Documents        []string
	TopN             *int
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type RerankResult struct {
	OriginalDocuments []string
	RerankedDocuments []string
	Ranking           []RerankingResult
	Results           []RerankingResult
	Usage             Usage
	Warnings          []Warning
	ProviderMetadata  ProviderMetadata
	Response          ResponseMetadata
}

func Rerank(ctx context.Context, opts RerankOptions) (rerankResult *RerankResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventRerankStart, OperationRerank, opts.Model, map[string]any{
		"query":          opts.Query,
		"document_count": len(opts.Documents),
		"top_n":          opts.TopN,
	})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventRerankError, OperationRerank, err)
		}
	}()
	if opts.Model == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "model is required"}
	}
	result, err := opts.Model.DoRerank(ctx, RerankingModelCallOptions{
		Query:           opts.Query,
		Documents:       opts.Documents,
		TopN:            opts.TopN,
		Headers:         withUserAgent(opts.Headers, "go-ai/"+Version),
		ProviderOptions: opts.ProviderOptions,
	})
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "model returned nil rerank result"}
	}
	ranking, rerankedDocuments, err := normalizeRerankingResults(opts.Documents, result.Results)
	if err != nil {
		return nil, err
	}
	LogWarnings(result.Warnings, opts.Model.Provider(), opts.Model.ModelID())
	rerankResult = &RerankResult{
		OriginalDocuments: append([]string{}, opts.Documents...),
		RerankedDocuments: rerankedDocuments,
		Ranking:           ranking,
		Results:           result.Results,
		Usage:             result.Usage,
		Warnings:          result.Warnings,
		ProviderMetadata:  result.ProviderMetadata,
		Response:          result.Response,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventRerankFinish, OperationRerank, rerankResult, map[string]any{
		"result_count": len(rerankResult.Ranking),
		"usage":        rerankResult.Usage,
	})
	return rerankResult, nil
}

func normalizeRerankingResults(documents []string, results []RerankingResult) ([]RerankingResult, []string, error) {
	ranking := make([]RerankingResult, len(results))
	rerankedDocuments := make([]string, len(results))
	for i, result := range results {
		if result.Index < 0 || result.Index >= len(documents) {
			return nil, nil, &SDKError{Kind: ErrInvalidResponseData, Message: "rerank result index is out of range"}
		}
		if result.Document == "" {
			result.Document = documents[result.Index]
		}
		ranking[i] = result
		rerankedDocuments[i] = result.Document
	}
	return ranking, rerankedDocuments, nil
}

type wrappedVideoModel struct {
	model      VideoModel
	middleware []VideoModelMiddleware
}

func (m wrappedVideoModel) Provider() string {
	for _, middleware := range m.middleware {
		if provider := middleware.OverrideVideoProvider(m.model); provider != "" {
			return provider
		}
	}
	return m.model.Provider()
}

func (m wrappedVideoModel) ModelID() string {
	for _, middleware := range m.middleware {
		if modelID := middleware.OverrideVideoModelID(m.model); modelID != "" {
			return modelID
		}
	}
	return m.model.ModelID()
}

func (m wrappedVideoModel) DoGenerateVideo(ctx context.Context, opts VideoModelCallOptions) (*VideoModelResult, error) {
	var next VideoGenerateMiddlewareNext
	index := 0
	next = func(ctx context.Context, opts VideoModelCallOptions) (*VideoModelResult, error) {
		if index >= len(m.middleware) {
			return m.model.DoGenerateVideo(ctx, opts)
		}
		current := m.middleware[index]
		index++
		return current.WrapGenerateVideo(ctx, m.model, opts, next)
	}
	return next(ctx, opts)
}

type wrappedSpeechModel struct {
	model      SpeechModel
	middleware []SpeechModelMiddleware
}

func (m wrappedSpeechModel) Provider() string {
	for _, middleware := range m.middleware {
		if provider := middleware.OverrideSpeechProvider(m.model); provider != "" {
			return provider
		}
	}
	return m.model.Provider()
}

func (m wrappedSpeechModel) ModelID() string {
	for _, middleware := range m.middleware {
		if modelID := middleware.OverrideSpeechModelID(m.model); modelID != "" {
			return modelID
		}
	}
	return m.model.ModelID()
}

func (m wrappedSpeechModel) DoGenerateSpeech(ctx context.Context, opts SpeechModelCallOptions) (*SpeechModelResult, error) {
	var next SpeechGenerateMiddlewareNext
	index := 0
	next = func(ctx context.Context, opts SpeechModelCallOptions) (*SpeechModelResult, error) {
		if index >= len(m.middleware) {
			return m.model.DoGenerateSpeech(ctx, opts)
		}
		current := m.middleware[index]
		index++
		return current.WrapGenerateSpeech(ctx, m.model, opts, next)
	}
	return next(ctx, opts)
}

type wrappedTranscriptionModel struct {
	model      TranscriptionModel
	middleware []TranscriptionModelMiddleware
}

func (m wrappedTranscriptionModel) Provider() string {
	for _, middleware := range m.middleware {
		if provider := middleware.OverrideTranscriptionProvider(m.model); provider != "" {
			return provider
		}
	}
	return m.model.Provider()
}

func (m wrappedTranscriptionModel) ModelID() string {
	for _, middleware := range m.middleware {
		if modelID := middleware.OverrideTranscriptionModelID(m.model); modelID != "" {
			return modelID
		}
	}
	return m.model.ModelID()
}

func (m wrappedTranscriptionModel) DoTranscribe(ctx context.Context, opts TranscriptionModelCallOptions) (*TranscriptionModelResult, error) {
	var next TranscriptionMiddlewareNext
	index := 0
	next = func(ctx context.Context, opts TranscriptionModelCallOptions) (*TranscriptionModelResult, error) {
		if index >= len(m.middleware) {
			return m.model.DoTranscribe(ctx, opts)
		}
		current := m.middleware[index]
		index++
		return current.WrapTranscribe(ctx, m.model, opts, next)
	}
	return next(ctx, opts)
}

type wrappedRerankingModel struct {
	model      RerankingModel
	middleware []RerankingModelMiddleware
}

func (m wrappedRerankingModel) Provider() string {
	for _, middleware := range m.middleware {
		if provider := middleware.OverrideRerankingProvider(m.model); provider != "" {
			return provider
		}
	}
	return m.model.Provider()
}

func (m wrappedRerankingModel) ModelID() string {
	for _, middleware := range m.middleware {
		if modelID := middleware.OverrideRerankingModelID(m.model); modelID != "" {
			return modelID
		}
	}
	return m.model.ModelID()
}

func (m wrappedRerankingModel) DoRerank(ctx context.Context, opts RerankingModelCallOptions) (*RerankingModelResult, error) {
	var next RerankMiddlewareNext
	index := 0
	next = func(ctx context.Context, opts RerankingModelCallOptions) (*RerankingModelResult, error) {
		if index >= len(m.middleware) {
			return m.model.DoRerank(ctx, opts)
		}
		current := m.middleware[index]
		index++
		return current.WrapRerank(ctx, m.model, opts, next)
	}
	return next(ctx, opts)
}
