package ai

import (
	"errors"
	"testing"
)

func TestProviderRegistryReturnsNoSuchModelWhenReferenceIsMalformed(t *testing.T) {
	registry := NewProviderRegistry(nil)
	_, err := registry.LanguageModel("model")
	if !errors.Is(err, ErrNoSuchModel) {
		t.Fatalf("expected no such model error, got %v", err)
	}
}

func TestProviderRegistrySupportsCustomSeparatorAndAdditionalSeparatorsInModelID(t *testing.T) {
	model := NewMockLanguageModel("model:part2")
	registry := NewProviderRegistryWithOptions(map[string]Provider{
		"mock": CustomProvider{LanguageModels: map[string]LanguageModel{"model:part2": model}},
	}, ProviderRegistryOptions{Separator: "|"})

	resolved, err := registry.LanguageModel("mock|model:part2")
	if err != nil {
		t.Fatal(err)
	}
	if resolved != model {
		t.Fatalf("expected model with additional colon to resolve")
	}
}

func TestCustomProviderUsesFallbackProvider(t *testing.T) {
	fallbackModel := NewMockLanguageModel("fallback")
	provider := CustomProvider{
		FallbackProvider: CustomProvider{
			LanguageModels: map[string]LanguageModel{"fallback": fallbackModel},
		},
	}

	if got := provider.LanguageModel("fallback"); got != fallbackModel {
		t.Fatalf("fallback language model = %#v", got)
	}
}

func TestProviderRegistryReturnsNoSuchModelForNilProviderResult(t *testing.T) {
	registry := NewProviderRegistry(map[string]Provider{
		"mock": CustomProvider{},
	})

	_, err := registry.ImageModel("mock:image")
	if !errors.Is(err, ErrNoSuchModel) {
		t.Fatalf("expected no such model error, got %v", err)
	}
}
