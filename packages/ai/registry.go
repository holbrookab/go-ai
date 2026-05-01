package ai

import (
	"fmt"
	"strings"
	"sync"
)

type ProviderRegistry struct {
	mu        sync.RWMutex
	providers map[string]Provider
	separator string
}

type ProviderRegistryOptions struct {
	Separator string
}

func NewProviderRegistry(providers map[string]Provider) *ProviderRegistry {
	return NewProviderRegistryWithOptions(providers, ProviderRegistryOptions{})
}

func NewProviderRegistryWithOptions(providers map[string]Provider, opts ProviderRegistryOptions) *ProviderRegistry {
	separator := opts.Separator
	if separator == "" {
		separator = ":"
	}
	registry := &ProviderRegistry{providers: map[string]Provider{}, separator: separator}
	for name, provider := range providers {
		registry.providers[name] = provider
	}
	return registry
}

func (r *ProviderRegistry) RegisterProvider(name string, provider Provider) error {
	if strings.TrimSpace(name) == "" {
		return &SDKError{Kind: ErrInvalidArgument, Message: "provider name is required"}
	}
	if provider == nil {
		return &SDKError{Kind: ErrInvalidArgument, Message: "provider is required"}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.providers == nil {
		r.providers = map[string]Provider{}
	}
	r.providers[name] = provider
	return nil
}

func (r *ProviderRegistry) LanguageModel(ref string) (LanguageModel, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	model := provider.LanguageModel(modelID)
	if isNil(model) {
		return nil, noSuchModelError("language model", ref)
	}
	return model, nil
}

func (r *ProviderRegistry) EmbeddingModel(ref string) (EmbeddingModel, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	embeddingProvider, ok := provider.(EmbeddingProvider)
	if !ok {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: fmt.Sprintf("provider %q does not support embeddings", providerName)}
	}
	model := embeddingProvider.EmbeddingModel(modelID)
	if isNil(model) {
		return nil, noSuchModelError("embedding model", ref)
	}
	return model, nil
}

func (r *ProviderRegistry) ImageModel(ref string) (ImageModel, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	imageProvider, ok := provider.(ImageProvider)
	if !ok {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: fmt.Sprintf("provider %q does not support images", providerName)}
	}
	model := imageProvider.ImageModel(modelID)
	if isNil(model) {
		return nil, noSuchModelError("image model", ref)
	}
	return model, nil
}

func (r *ProviderRegistry) VideoModel(ref string) (VideoModel, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	videoProvider, ok := provider.(VideoProvider)
	if !ok {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: fmt.Sprintf("provider %q does not support videos", providerName)}
	}
	model := videoProvider.VideoModel(modelID)
	if isNil(model) {
		return nil, noSuchModelError("video model", ref)
	}
	return model, nil
}

func (r *ProviderRegistry) SpeechModel(ref string) (SpeechModel, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	speechProvider, ok := provider.(SpeechProvider)
	if !ok {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: fmt.Sprintf("provider %q does not support speech", providerName)}
	}
	model := speechProvider.SpeechModel(modelID)
	if isNil(model) {
		return nil, noSuchModelError("speech model", ref)
	}
	return model, nil
}

func (r *ProviderRegistry) TranscriptionModel(ref string) (TranscriptionModel, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	transcriptionProvider, ok := provider.(TranscriptionProvider)
	if !ok {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: fmt.Sprintf("provider %q does not support transcription", providerName)}
	}
	model := transcriptionProvider.TranscriptionModel(modelID)
	if isNil(model) {
		return nil, noSuchModelError("transcription model", ref)
	}
	return model, nil
}

func (r *ProviderRegistry) RerankingModel(ref string) (RerankingModel, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	rerankingProvider, ok := provider.(RerankingProvider)
	if !ok {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: fmt.Sprintf("provider %q does not support reranking", providerName)}
	}
	model := rerankingProvider.RerankingModel(modelID)
	if isNil(model) {
		return nil, noSuchModelError("reranking model", ref)
	}
	return model, nil
}

func splitProviderModelRef(ref string) (string, string, error) {
	parts := strings.SplitN(ref, ":", 2)
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
		return "", "", noSuchModelError("model", ref)
	}
	return parts[0], parts[1], nil
}

func (r *ProviderRegistry) splitProviderModelRef(ref string) (string, string, error) {
	separator := r.separator
	if separator == "" {
		separator = ":"
	}
	parts := strings.SplitN(ref, separator, 2)
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
		return "", "", noSuchModelError("model", ref)
	}
	return parts[0], parts[1], nil
}

type CustomProvider struct {
	LanguageModels      map[string]LanguageModel
	EmbeddingModels     map[string]EmbeddingModel
	ImageModels         map[string]ImageModel
	VideoModels         map[string]VideoModel
	SpeechModels        map[string]SpeechModel
	TranscriptionModels map[string]TranscriptionModel
	RerankingModels     map[string]RerankingModel
	FallbackProvider    Provider
}

func (p CustomProvider) LanguageModel(modelID string) LanguageModel {
	if model := p.LanguageModels[modelID]; !isNil(model) {
		return model
	}
	if p.FallbackProvider != nil {
		return p.FallbackProvider.LanguageModel(modelID)
	}
	return nil
}

func (p CustomProvider) EmbeddingModel(modelID string) EmbeddingModel {
	if model := p.EmbeddingModels[modelID]; !isNil(model) {
		return model
	}
	if provider, ok := p.FallbackProvider.(EmbeddingProvider); ok {
		return provider.EmbeddingModel(modelID)
	}
	return nil
}

func (p CustomProvider) ImageModel(modelID string) ImageModel {
	if model := p.ImageModels[modelID]; !isNil(model) {
		return model
	}
	if provider, ok := p.FallbackProvider.(ImageProvider); ok {
		return provider.ImageModel(modelID)
	}
	return nil
}

func (p CustomProvider) VideoModel(modelID string) VideoModel {
	if model := p.VideoModels[modelID]; !isNil(model) {
		return model
	}
	if provider, ok := p.FallbackProvider.(VideoProvider); ok {
		return provider.VideoModel(modelID)
	}
	return nil
}

func (p CustomProvider) SpeechModel(modelID string) SpeechModel {
	if model := p.SpeechModels[modelID]; !isNil(model) {
		return model
	}
	if provider, ok := p.FallbackProvider.(SpeechProvider); ok {
		return provider.SpeechModel(modelID)
	}
	return nil
}

func (p CustomProvider) TranscriptionModel(modelID string) TranscriptionModel {
	if model := p.TranscriptionModels[modelID]; !isNil(model) {
		return model
	}
	if provider, ok := p.FallbackProvider.(TranscriptionProvider); ok {
		return provider.TranscriptionModel(modelID)
	}
	return nil
}

func (p CustomProvider) RerankingModel(modelID string) RerankingModel {
	if model := p.RerankingModels[modelID]; !isNil(model) {
		return model
	}
	if provider, ok := p.FallbackProvider.(RerankingProvider); ok {
		return provider.RerankingModel(modelID)
	}
	return nil
}
