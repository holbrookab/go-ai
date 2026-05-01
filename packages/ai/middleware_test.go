package ai

import (
	"context"
	"reflect"
	"testing"
)

func TestWrapLanguageModelMiddlewareOrder(t *testing.T) {
	model := NewMockLanguageModel("lm")
	var order []string
	wrapped := WrapLanguageModel(
		model,
		MiddlewareFunc{Generate: func(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next GenerateMiddlewareNext) (*LanguageModelGenerateResult, error) {
			order = append(order, "first-before")
			result, err := next(ctx, opts)
			order = append(order, "first-after")
			return result, err
		}},
		MiddlewareFunc{Generate: func(ctx context.Context, model LanguageModel, opts LanguageModelCallOptions, next GenerateMiddlewareNext) (*LanguageModelGenerateResult, error) {
			order = append(order, "second-before")
			result, err := next(ctx, opts)
			order = append(order, "second-after")
			return result, err
		}},
	)

	if _, err := wrapped.DoGenerate(context.Background(), LanguageModelCallOptions{}); err != nil {
		t.Fatal(err)
	}

	want := []string{"first-before", "second-before", "second-after", "first-after"}
	if !reflect.DeepEqual(order, want) {
		t.Fatalf("order = %#v, want %#v", order, want)
	}
}

func TestWrapEmbeddingModelAppliesMiddlewareAndOverrides(t *testing.T) {
	model := NewMockEmbeddingModel("embedding")
	model.MaxPerCall = 4
	wrapped := WrapEmbeddingModel(model, EmbeddingMiddlewareFunc{
		Provider: func(EmbeddingModel) string { return "wrapped" },
		ModelID:  func(EmbeddingModel) string { return "wrapped-embedding" },
		MaxEmbeddingsPerCallOverride: func(EmbeddingModel) int {
			return 8
		},
		Embed: func(ctx context.Context, model EmbeddingModel, opts EmbeddingModelCallOptions, next EmbedMiddlewareNext) (*EmbeddingModelResult, error) {
			opts.Values = append(opts.Values, "from-middleware")
			return next(ctx, opts)
		},
	})

	if wrapped.Provider() != "wrapped" {
		t.Fatalf("provider = %q", wrapped.Provider())
	}
	if wrapped.ModelID() != "wrapped-embedding" {
		t.Fatalf("model id = %q", wrapped.ModelID())
	}
	if wrapped.MaxEmbeddingsPerCall() != 8 {
		t.Fatalf("max = %d", wrapped.MaxEmbeddingsPerCall())
	}
	if _, err := wrapped.DoEmbed(context.Background(), EmbeddingModelCallOptions{Values: []string{"input"}}); err != nil {
		t.Fatal(err)
	}
	if got := model.Calls[0].Values; !reflect.DeepEqual(got, []string{"input", "from-middleware"}) {
		t.Fatalf("values = %#v", got)
	}
}

func TestDefaultEmbeddingSettingsMergesHeadersAndProviderOptions(t *testing.T) {
	model := NewMockEmbeddingModel("embedding")
	wrapped := WrapEmbeddingModel(model, DefaultEmbeddingSettings(EmbeddingModelCallOptions{
		Headers: map[string]string{"x-default": "yes", "x-shared": "default"},
		ProviderOptions: ProviderOptions{
			"provider": map[string]any{"default": true, "shared": "default"},
		},
	}))

	_, err := wrapped.DoEmbed(context.Background(), EmbeddingModelCallOptions{
		Values:  []string{"input"},
		Headers: map[string]string{"x-shared": "call"},
		ProviderOptions: ProviderOptions{
			"provider": map[string]any{"call": true, "shared": "call"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	got := model.Calls[0]
	if got.Headers["x-default"] != "yes" || got.Headers["x-shared"] != "call" {
		t.Fatalf("headers = %#v", got.Headers)
	}
	providerOptions := got.ProviderOptions["provider"].(map[string]any)
	if providerOptions["default"] != true || providerOptions["call"] != true || providerOptions["shared"] != "call" {
		t.Fatalf("provider options = %#v", providerOptions)
	}
}

func TestDefaultSettingsMergesHeadersAndNestedProviderOptions(t *testing.T) {
	model := NewMockLanguageModel("lm")
	wrapped := WrapLanguageModel(model, DefaultSettings(LanguageModelCallOptions{
		Headers: map[string]string{"x-default": "yes", "x-shared": "default"},
		ProviderOptions: ProviderOptions{
			"anthropic": map[string]any{
				"cacheControl": map[string]any{"type": "ephemeral"},
				"tools":        map[string]any{"math": map[string]any{"enabled": true}},
			},
		},
	}))

	_, err := wrapped.DoGenerate(context.Background(), LanguageModelCallOptions{
		Headers: map[string]string{"x-shared": "call"},
		ProviderOptions: ProviderOptions{
			"anthropic": map[string]any{
				"tools": map[string]any{
					"retrieval": map[string]any{"enabled": false},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	got := model.GenerateCalls[0]
	if got.Headers["x-default"] != "yes" || got.Headers["x-shared"] != "call" {
		t.Fatalf("headers = %#v", got.Headers)
	}
	anthropic := got.ProviderOptions["anthropic"].(map[string]any)
	tools := anthropic["tools"].(map[string]any)
	if anthropic["cacheControl"] == nil || tools["math"] == nil || tools["retrieval"] == nil {
		t.Fatalf("provider options = %#v", got.ProviderOptions)
	}
}

func TestWrapProviderWrapsLanguageEmbeddingAndImageModels(t *testing.T) {
	provider := CustomProvider{
		LanguageModels:  map[string]LanguageModel{"lm": NewMockLanguageModel("lm")},
		EmbeddingModels: map[string]EmbeddingModel{"embed": NewMockEmbeddingModel("embed")},
		ImageModels:     map[string]ImageModel{"image": NewMockImageModel("image")},
	}
	wrapped := WrapProviderWithEmbedding(
		provider,
		[]LanguageModelMiddleware{DefaultSettings(LanguageModelCallOptions{Reasoning: "trace"})},
		[]EmbeddingModelMiddleware{DefaultEmbeddingSettings(EmbeddingModelCallOptions{Headers: map[string]string{"x-embed": "yes"}})},
		[]ImageModelMiddleware{ImageMiddlewareFunc{Provider: func(ImageModel) string { return "wrapped-image" }}},
	)

	language := wrapped.LanguageModel("lm")
	if _, err := language.DoGenerate(context.Background(), LanguageModelCallOptions{}); err != nil {
		t.Fatal(err)
	}
	if provider.LanguageModels["lm"].(*MockLanguageModel).GenerateCalls[0].Reasoning != "trace" {
		t.Fatalf("language middleware was not applied")
	}

	embedding := wrapped.EmbeddingModel("embed")
	if _, err := embedding.DoEmbed(context.Background(), EmbeddingModelCallOptions{Values: []string{"value"}}); err != nil {
		t.Fatal(err)
	}
	if provider.EmbeddingModels["embed"].(*MockEmbeddingModel).Calls[0].Headers["x-embed"] != "yes" {
		t.Fatalf("embedding middleware was not applied")
	}

	image := wrapped.ImageModel("image")
	if image.Provider() != "wrapped-image" {
		t.Fatalf("image provider = %q", image.Provider())
	}
}

func TestMediaModelWrappersApplyMiddlewareAndOverrides(t *testing.T) {
	ctx := context.Background()

	video := NewMockVideoModel("video")
	wrappedVideo := WrapVideoModel(video, VideoMiddlewareFunc{
		Provider: func(VideoModel) string { return "wrapped-video" },
		ModelID:  func(VideoModel) string { return "wrapped-video-id" },
		Generate: func(ctx context.Context, model VideoModel, opts VideoModelCallOptions, next VideoGenerateMiddlewareNext) (*VideoModelResult, error) {
			opts.Prompt += " from middleware"
			return next(ctx, opts)
		},
	})
	if wrappedVideo.Provider() != "wrapped-video" || wrappedVideo.ModelID() != "wrapped-video-id" {
		t.Fatalf("video overrides = %q/%q", wrappedVideo.Provider(), wrappedVideo.ModelID())
	}
	if _, err := wrappedVideo.DoGenerateVideo(ctx, VideoModelCallOptions{Prompt: "clip"}); err != nil {
		t.Fatal(err)
	}
	if video.Calls[0].Prompt != "clip from middleware" {
		t.Fatalf("video prompt = %q", video.Calls[0].Prompt)
	}

	speech := NewMockSpeechModel("speech")
	wrappedSpeech := WrapSpeechModel(speech, SpeechMiddlewareFunc{
		Provider: func(SpeechModel) string { return "wrapped-speech" },
		Generate: func(ctx context.Context, model SpeechModel, opts SpeechModelCallOptions, next SpeechGenerateMiddlewareNext) (*SpeechModelResult, error) {
			opts.Text += " spoken"
			return next(ctx, opts)
		},
	})
	if wrappedSpeech.Provider() != "wrapped-speech" {
		t.Fatalf("speech provider = %q", wrappedSpeech.Provider())
	}
	if _, err := wrappedSpeech.DoGenerateSpeech(ctx, SpeechModelCallOptions{Text: "hello"}); err != nil {
		t.Fatal(err)
	}
	if speech.Calls[0].Text != "hello spoken" {
		t.Fatalf("speech text = %q", speech.Calls[0].Text)
	}

	transcription := NewMockTranscriptionModel("transcription")
	wrappedTranscription := WrapTranscriptionModel(transcription, TranscriptionMiddlewareFunc{
		ModelID: func(TranscriptionModel) string { return "wrapped-transcription" },
		Transcribe: func(ctx context.Context, model TranscriptionModel, opts TranscriptionModelCallOptions, next TranscriptionMiddlewareNext) (*TranscriptionModelResult, error) {
			opts.Prompt = "transcribe this"
			return next(ctx, opts)
		},
	})
	if wrappedTranscription.ModelID() != "wrapped-transcription" {
		t.Fatalf("transcription model id = %q", wrappedTranscription.ModelID())
	}
	if _, err := wrappedTranscription.DoTranscribe(ctx, TranscriptionModelCallOptions{}); err != nil {
		t.Fatal(err)
	}
	if transcription.Calls[0].Prompt != "transcribe this" {
		t.Fatalf("transcription prompt = %q", transcription.Calls[0].Prompt)
	}

	reranking := NewMockRerankingModel("reranking")
	wrappedReranking := WrapRerankingModel(reranking, RerankingMiddlewareFunc{
		Provider: func(RerankingModel) string { return "wrapped-reranking" },
		Rerank: func(ctx context.Context, model RerankingModel, opts RerankingModelCallOptions, next RerankMiddlewareNext) (*RerankingModelResult, error) {
			opts.Query += " query"
			return next(ctx, opts)
		},
	})
	if wrappedReranking.Provider() != "wrapped-reranking" {
		t.Fatalf("reranking provider = %q", wrappedReranking.Provider())
	}
	if _, err := wrappedReranking.DoRerank(ctx, RerankingModelCallOptions{Query: "search"}); err != nil {
		t.Fatal(err)
	}
	if reranking.Calls[0].Query != "search query" {
		t.Fatalf("reranking query = %q", reranking.Calls[0].Query)
	}
}
