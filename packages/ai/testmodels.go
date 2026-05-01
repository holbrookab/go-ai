package ai

import (
	"context"
	"sync"
)

type MockProvider struct {
	LanguageModels      map[string]LanguageModel
	EmbeddingModels     map[string]EmbeddingModel
	ImageModels         map[string]ImageModel
	VideoModels         map[string]VideoModel
	SpeechModels        map[string]SpeechModel
	TranscriptionModels map[string]TranscriptionModel
	RerankingModels     map[string]RerankingModel
	FilesAPIValue       FilesAPI
	SkillsAPIValue      SkillsAPI
}

func NewMockProvider() *MockProvider {
	return &MockProvider{
		LanguageModels:      map[string]LanguageModel{},
		EmbeddingModels:     map[string]EmbeddingModel{},
		ImageModels:         map[string]ImageModel{},
		VideoModels:         map[string]VideoModel{},
		SpeechModels:        map[string]SpeechModel{},
		TranscriptionModels: map[string]TranscriptionModel{},
		RerankingModels:     map[string]RerankingModel{},
	}
}

func (p *MockProvider) LanguageModel(modelID string) LanguageModel {
	return p.LanguageModels[modelID]
}

func (p *MockProvider) EmbeddingModel(modelID string) EmbeddingModel {
	return p.EmbeddingModels[modelID]
}

func (p *MockProvider) ImageModel(modelID string) ImageModel {
	return p.ImageModels[modelID]
}

func (p *MockProvider) VideoModel(modelID string) VideoModel {
	return p.VideoModels[modelID]
}

func (p *MockProvider) SpeechModel(modelID string) SpeechModel {
	return p.SpeechModels[modelID]
}

func (p *MockProvider) TranscriptionModel(modelID string) TranscriptionModel {
	return p.TranscriptionModels[modelID]
}

func (p *MockProvider) RerankingModel(modelID string) RerankingModel {
	return p.RerankingModels[modelID]
}

func (p *MockProvider) Files() FilesAPI {
	return p.FilesAPIValue
}

func (p *MockProvider) Skills() SkillsAPI {
	return p.SkillsAPIValue
}

type MockLanguageModel struct {
	ProviderName string
	ID           string

	SupportedURLsFunc func(context.Context) (map[string][]string, error)
	GenerateFunc      func(context.Context, LanguageModelCallOptions) (*LanguageModelGenerateResult, error)
	StreamFunc        func(context.Context, LanguageModelCallOptions) (*LanguageModelStreamResult, error)

	mu            sync.Mutex
	GenerateCalls []LanguageModelCallOptions
	StreamCalls   []LanguageModelCallOptions
}

func NewMockLanguageModel(id string) *MockLanguageModel {
	return &MockLanguageModel{ProviderName: "mock", ID: id}
}

func (m *MockLanguageModel) Provider() string {
	if m.ProviderName == "" {
		return "mock"
	}
	return m.ProviderName
}

func (m *MockLanguageModel) ModelID() string {
	if m.ID == "" {
		return "mock-model"
	}
	return m.ID
}

func (m *MockLanguageModel) SupportedURLs(ctx context.Context) (map[string][]string, error) {
	if m.SupportedURLsFunc != nil {
		return m.SupportedURLsFunc(ctx)
	}
	return nil, nil
}

func (m *MockLanguageModel) DoGenerate(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
	m.mu.Lock()
	m.GenerateCalls = append(m.GenerateCalls, opts)
	m.mu.Unlock()
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, opts)
	}
	return &LanguageModelGenerateResult{
		Content:      []Part{TextPart{Text: "ok"}},
		FinishReason: FinishReason{Unified: FinishStop, Raw: FinishStop},
	}, nil
}

func (m *MockLanguageModel) DoStream(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
	m.mu.Lock()
	m.StreamCalls = append(m.StreamCalls, opts)
	m.mu.Unlock()
	if m.StreamFunc != nil {
		return m.StreamFunc(ctx, opts)
	}
	ch := make(chan StreamPart)
	close(ch)
	return &LanguageModelStreamResult{Stream: ch}, nil
}

type MockEmbeddingModel struct {
	ProviderName string
	ID           string
	MaxPerCall   int

	EmbedFunc func(context.Context, EmbeddingModelCallOptions) (*EmbeddingModelResult, error)

	mu    sync.Mutex
	Calls []EmbeddingModelCallOptions
}

func NewMockEmbeddingModel(id string) *MockEmbeddingModel {
	return &MockEmbeddingModel{ProviderName: "mock", ID: id}
}

func (m *MockEmbeddingModel) Provider() string {
	if m.ProviderName == "" {
		return "mock"
	}
	return m.ProviderName
}

func (m *MockEmbeddingModel) ModelID() string {
	if m.ID == "" {
		return "mock-embedding-model"
	}
	return m.ID
}

func (m *MockEmbeddingModel) MaxEmbeddingsPerCall() int {
	return m.MaxPerCall
}

func (m *MockEmbeddingModel) DoEmbed(ctx context.Context, opts EmbeddingModelCallOptions) (*EmbeddingModelResult, error) {
	m.mu.Lock()
	m.Calls = append(m.Calls, opts)
	m.mu.Unlock()
	if m.EmbedFunc != nil {
		return m.EmbedFunc(ctx, opts)
	}
	embeddings := make([][]float64, len(opts.Values))
	for i := range opts.Values {
		embeddings[i] = []float64{float64(i)}
	}
	return &EmbeddingModelResult{Embeddings: embeddings}, nil
}

type MockImageModel struct {
	ProviderName string
	ID           string

	GenerateFunc func(context.Context, ImageModelCallOptions) (*ImageModelResult, error)

	mu    sync.Mutex
	Calls []ImageModelCallOptions
}

func NewMockImageModel(id string) *MockImageModel {
	return &MockImageModel{ProviderName: "mock", ID: id}
}

func (m *MockImageModel) Provider() string {
	if m.ProviderName == "" {
		return "mock"
	}
	return m.ProviderName
}

func (m *MockImageModel) ModelID() string {
	if m.ID == "" {
		return "mock-image-model"
	}
	return m.ID
}

func (m *MockImageModel) DoGenerateImage(ctx context.Context, opts ImageModelCallOptions) (*ImageModelResult, error) {
	m.mu.Lock()
	m.Calls = append(m.Calls, opts)
	m.mu.Unlock()
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, opts)
	}
	return &ImageModelResult{Images: []GeneratedFile{{Data: []byte("image"), MediaType: "image/png"}}}, nil
}

type MockVideoModel struct {
	ProviderName string
	ID           string

	GenerateFunc func(context.Context, VideoModelCallOptions) (*VideoModelResult, error)

	mu    sync.Mutex
	Calls []VideoModelCallOptions
}

func NewMockVideoModel(id string) *MockVideoModel {
	return &MockVideoModel{ProviderName: "mock", ID: id}
}

func (m *MockVideoModel) Provider() string {
	if m.ProviderName == "" {
		return "mock"
	}
	return m.ProviderName
}

func (m *MockVideoModel) ModelID() string {
	if m.ID == "" {
		return "mock-video-model"
	}
	return m.ID
}

func (m *MockVideoModel) DoGenerateVideo(ctx context.Context, opts VideoModelCallOptions) (*VideoModelResult, error) {
	m.mu.Lock()
	m.Calls = append(m.Calls, opts)
	m.mu.Unlock()
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, opts)
	}
	return &VideoModelResult{Videos: []GeneratedFile{{Data: []byte("video"), MediaType: "video/mp4"}}}, nil
}

type MockSpeechModel struct {
	ProviderName string
	ID           string

	GenerateFunc func(context.Context, SpeechModelCallOptions) (*SpeechModelResult, error)

	mu    sync.Mutex
	Calls []SpeechModelCallOptions
}

func NewMockSpeechModel(id string) *MockSpeechModel {
	return &MockSpeechModel{ProviderName: "mock", ID: id}
}

func (m *MockSpeechModel) Provider() string {
	if m.ProviderName == "" {
		return "mock"
	}
	return m.ProviderName
}

func (m *MockSpeechModel) ModelID() string {
	if m.ID == "" {
		return "mock-speech-model"
	}
	return m.ID
}

func (m *MockSpeechModel) DoGenerateSpeech(ctx context.Context, opts SpeechModelCallOptions) (*SpeechModelResult, error) {
	m.mu.Lock()
	m.Calls = append(m.Calls, opts)
	m.mu.Unlock()
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, opts)
	}
	return &SpeechModelResult{Audio: GeneratedFile{Data: []byte("audio"), MediaType: "audio/mpeg"}}, nil
}

type MockTranscriptionModel struct {
	ProviderName string
	ID           string

	TranscribeFunc func(context.Context, TranscriptionModelCallOptions) (*TranscriptionModelResult, error)

	mu    sync.Mutex
	Calls []TranscriptionModelCallOptions
}

func NewMockTranscriptionModel(id string) *MockTranscriptionModel {
	return &MockTranscriptionModel{ProviderName: "mock", ID: id}
}

func (m *MockTranscriptionModel) Provider() string {
	if m.ProviderName == "" {
		return "mock"
	}
	return m.ProviderName
}

func (m *MockTranscriptionModel) ModelID() string {
	if m.ID == "" {
		return "mock-transcription-model"
	}
	return m.ID
}

func (m *MockTranscriptionModel) DoTranscribe(ctx context.Context, opts TranscriptionModelCallOptions) (*TranscriptionModelResult, error) {
	m.mu.Lock()
	m.Calls = append(m.Calls, opts)
	m.mu.Unlock()
	if m.TranscribeFunc != nil {
		return m.TranscribeFunc(ctx, opts)
	}
	return &TranscriptionModelResult{Text: "transcript"}, nil
}

type MockRerankingModel struct {
	ProviderName string
	ID           string

	RerankFunc func(context.Context, RerankingModelCallOptions) (*RerankingModelResult, error)

	mu    sync.Mutex
	Calls []RerankingModelCallOptions
}

func NewMockRerankingModel(id string) *MockRerankingModel {
	return &MockRerankingModel{ProviderName: "mock", ID: id}
}

func (m *MockRerankingModel) Provider() string {
	if m.ProviderName == "" {
		return "mock"
	}
	return m.ProviderName
}

func (m *MockRerankingModel) ModelID() string {
	if m.ID == "" {
		return "mock-reranking-model"
	}
	return m.ID
}

func (m *MockRerankingModel) DoRerank(ctx context.Context, opts RerankingModelCallOptions) (*RerankingModelResult, error) {
	m.mu.Lock()
	m.Calls = append(m.Calls, opts)
	m.mu.Unlock()
	if m.RerankFunc != nil {
		return m.RerankFunc(ctx, opts)
	}
	results := make([]RerankingResult, len(opts.Documents))
	for i, doc := range opts.Documents {
		results[i] = RerankingResult{Index: i, Document: doc, Score: 1}
	}
	return &RerankingModelResult{Results: results}, nil
}
