package ai

import (
	"context"
	"testing"
)

func TestMockLanguageModelRecordsGenerateCalls(t *testing.T) {
	model := NewMockLanguageModel("test")
	_, err := model.DoGenerate(context.Background(), LanguageModelCallOptions{
		Prompt: []Message{UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if len(model.GenerateCalls) != 1 {
		t.Fatalf("expected generate call to be recorded")
	}
	if got := model.GenerateCalls[0].Prompt[0].Role; got != RoleUser {
		t.Fatalf("unexpected recorded prompt role %q", got)
	}
}

func TestMockProviderReturnsConfiguredModels(t *testing.T) {
	provider := NewMockProvider()
	language := NewMockLanguageModel("language")
	embedding := NewMockEmbeddingModel("embedding")
	image := NewMockImageModel("image")
	video := NewMockVideoModel("video")
	speech := NewMockSpeechModel("speech")
	transcription := NewMockTranscriptionModel("transcription")
	reranking := NewMockRerankingModel("reranking")
	provider.LanguageModels["language"] = language
	provider.EmbeddingModels["embedding"] = embedding
	provider.ImageModels["image"] = image
	provider.VideoModels["video"] = video
	provider.SpeechModels["speech"] = speech
	provider.TranscriptionModels["transcription"] = transcription
	provider.RerankingModels["reranking"] = reranking

	if provider.LanguageModel("language") != language ||
		provider.EmbeddingModel("embedding") != embedding ||
		provider.ImageModel("image") != image ||
		provider.VideoModel("video") != video ||
		provider.SpeechModel("speech") != speech ||
		provider.TranscriptionModel("transcription") != transcription ||
		provider.RerankingModel("reranking") != reranking {
		t.Fatalf("mock provider did not return configured models")
	}
}

func TestMockEmbeddingModelRecordsCalls(t *testing.T) {
	model := NewMockEmbeddingModel("test")
	result, err := model.DoEmbed(context.Background(), EmbeddingModelCallOptions{
		Values: []string{"a", "b"},
	})
	if err != nil {
		t.Fatalf("DoEmbed failed: %v", err)
	}
	if len(model.Calls) != 1 {
		t.Fatalf("expected embedding call to be recorded")
	}
	if len(result.Embeddings) != 2 {
		t.Fatalf("expected default embeddings for each input, got %d", len(result.Embeddings))
	}
}

func TestMockMediaModelsReturnUsableDefaults(t *testing.T) {
	ctx := context.Background()
	image, err := NewMockImageModel("image").DoGenerateImage(ctx, ImageModelCallOptions{})
	if err != nil || len(image.Images) != 1 {
		t.Fatalf("expected default image, got %#v, %v", image, err)
	}
	video, err := NewMockVideoModel("video").DoGenerateVideo(ctx, VideoModelCallOptions{})
	if err != nil || len(video.Videos) != 1 {
		t.Fatalf("expected default video, got %#v, %v", video, err)
	}
	speech, err := NewMockSpeechModel("speech").DoGenerateSpeech(ctx, SpeechModelCallOptions{})
	if err != nil || len(speech.Audio.Data) == 0 {
		t.Fatalf("expected default speech, got %#v, %v", speech, err)
	}
	transcription, err := NewMockTranscriptionModel("transcription").DoTranscribe(ctx, TranscriptionModelCallOptions{})
	if err != nil || transcription.Text == "" {
		t.Fatalf("expected default transcription, got %#v, %v", transcription, err)
	}
	rerank, err := NewMockRerankingModel("rerank").DoRerank(ctx, RerankingModelCallOptions{Documents: []string{"a", "b"}})
	if err != nil || len(rerank.Results) != 2 {
		t.Fatalf("expected default rerank results, got %#v, %v", rerank, err)
	}
}
