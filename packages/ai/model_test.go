package ai

import (
	"errors"
	"testing"
)

func TestResolveLanguageModelReturnsDirectModel(t *testing.T) {
	model := NewMockLanguageModel("direct")
	resolved, err := ResolveLanguageModel(model, nil)
	if err != nil {
		t.Fatalf("ResolveLanguageModel failed: %v", err)
	}
	if resolved != model {
		t.Fatalf("expected direct model to be returned")
	}
}

func TestResolveLanguageModelUsesRegistryReference(t *testing.T) {
	model := NewMockLanguageModel("resolved")
	registry := NewProviderRegistry(map[string]Provider{
		"mock": CustomProvider{LanguageModels: map[string]LanguageModel{"resolved": model}},
	})
	resolved, err := ResolveLanguageModel("mock:resolved", registry)
	if err != nil {
		t.Fatalf("ResolveLanguageModel failed: %v", err)
	}
	if resolved != model {
		t.Fatalf("expected registry model to be returned")
	}
}

func TestResolveLanguageModelReturnsNoSuchModelForCustomProviderNilModel(t *testing.T) {
	registry := NewProviderRegistry(map[string]Provider{
		"mock": CustomProvider{},
	})
	_, err := ResolveLanguageModel("mock:missing", registry)
	if !errors.Is(err, ErrNoSuchModel) {
		t.Fatalf("expected no such model error, got %v", err)
	}
}

func TestResolveLanguageModelPreservesNoSuchProvider(t *testing.T) {
	registry := NewProviderRegistry(nil)
	_, err := ResolveLanguageModel("missing:model", registry)
	if !errors.Is(err, ErrNoSuchProvider) {
		t.Fatalf("expected no such provider error, got %v", err)
	}
}

func TestResolveEmbeddingModelUsesRegistryReference(t *testing.T) {
	model := NewMockEmbeddingModel("embed")
	registry := NewProviderRegistry(map[string]Provider{
		"mock": CustomProvider{EmbeddingModels: map[string]EmbeddingModel{"embed": model}},
	})
	resolved, err := ResolveEmbeddingModel("mock:embed", registry)
	if err != nil {
		t.Fatalf("ResolveEmbeddingModel failed: %v", err)
	}
	if resolved != model {
		t.Fatalf("expected registry embedding model to be returned")
	}
}

func TestResolveEmbeddingModelReturnsNoSuchModelForCustomProviderNilModel(t *testing.T) {
	registry := NewProviderRegistry(map[string]Provider{
		"mock": CustomProvider{},
	})
	_, err := ResolveEmbeddingModel("mock:missing", registry)
	if !errors.Is(err, ErrNoSuchModel) {
		t.Fatalf("expected no such model error, got %v", err)
	}
}

func TestResolveMediaModelsUseRegistryReferences(t *testing.T) {
	image := NewMockImageModel("image")
	video := NewMockVideoModel("video")
	speech := NewMockSpeechModel("speech")
	transcription := NewMockTranscriptionModel("transcription")
	reranking := NewMockRerankingModel("reranking")
	registry := NewProviderRegistry(map[string]Provider{
		"mock": CustomProvider{
			ImageModels:         map[string]ImageModel{"image": image},
			VideoModels:         map[string]VideoModel{"video": video},
			SpeechModels:        map[string]SpeechModel{"speech": speech},
			TranscriptionModels: map[string]TranscriptionModel{"transcription": transcription},
			RerankingModels:     map[string]RerankingModel{"reranking": reranking},
		},
	})

	if resolved, err := ResolveImageModel("mock:image", registry); err != nil || resolved != image {
		t.Fatalf("ResolveImageModel = %#v, %v", resolved, err)
	}
	if resolved, err := ResolveVideoModel("mock:video", registry); err != nil || resolved != video {
		t.Fatalf("ResolveVideoModel = %#v, %v", resolved, err)
	}
	if resolved, err := ResolveSpeechModel("mock:speech", registry); err != nil || resolved != speech {
		t.Fatalf("ResolveSpeechModel = %#v, %v", resolved, err)
	}
	if resolved, err := ResolveTranscriptionModel("mock:transcription", registry); err != nil || resolved != transcription {
		t.Fatalf("ResolveTranscriptionModel = %#v, %v", resolved, err)
	}
	if resolved, err := ResolveRerankingModel("mock:reranking", registry); err != nil || resolved != reranking {
		t.Fatalf("ResolveRerankingModel = %#v, %v", resolved, err)
	}
}

func TestResolveMediaModelsReturnNoSuchModel(t *testing.T) {
	registry := NewProviderRegistry(map[string]Provider{
		"mock": CustomProvider{},
	})
	checks := []struct {
		name string
		run  func() error
	}{
		{"image", func() error { _, err := ResolveImageModel("mock:missing", registry); return err }},
		{"video", func() error { _, err := ResolveVideoModel("mock:missing", registry); return err }},
		{"speech", func() error { _, err := ResolveSpeechModel("mock:missing", registry); return err }},
		{"transcription", func() error { _, err := ResolveTranscriptionModel("mock:missing", registry); return err }},
		{"reranking", func() error { _, err := ResolveRerankingModel("mock:missing", registry); return err }},
	}
	for _, check := range checks {
		t.Run(check.name, func(t *testing.T) {
			if err := check.run(); !errors.Is(err, ErrNoSuchModel) {
				t.Fatalf("expected no such model error, got %v", err)
			}
		})
	}
}
