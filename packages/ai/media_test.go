package ai

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestGenerateImageLogsWarningsAndEmitsCallbacks(t *testing.T) {
	suppressUploadWarnings(t)
	model := NewMockImageModel("image")
	model.GenerateFunc = func(_ context.Context, _ ImageModelCallOptions) (*ImageModelResult, error) {
		return &ImageModelResult{
			Images:   []GeneratedFile{{Data: []byte("image"), MediaType: "image/png"}},
			Warnings: []Warning{{Type: "other", Message: "warning"}},
		}, nil
	}
	var calls []string
	result, err := GenerateImage(context.Background(), GenerateImageOptions{
		Model:  model,
		Prompt: "draw",
		OnStart: func(event StartEvent) {
			calls = append(calls, event.Operation+":"+event.Provider+":"+event.ModelID)
		},
		OnFinish: func(event FinishEvent) {
			calls = append(calls, event.Operation)
		},
	})
	if err != nil {
		t.Fatalf("GenerateImage failed: %v", err)
	}
	if len(result.Warnings) != 1 {
		t.Fatalf("expected warnings to be returned")
	}
	if !reflect.DeepEqual(calls, []string{"generate_image:mock:image", "generate_image"}) {
		t.Fatalf("unexpected callback calls: %#v", calls)
	}
}

func TestGenerateImageNoImagesUsesNamedError(t *testing.T) {
	model := NewMockImageModel("image")
	model.GenerateFunc = func(_ context.Context, _ ImageModelCallOptions) (*ImageModelResult, error) {
		return &ImageModelResult{Response: ResponseMetadata{ID: "response-1"}}, nil
	}
	_, err := GenerateImage(context.Background(), GenerateImageOptions{Model: model, Prompt: "draw"})
	if !IsNoImageGeneratedError(err) {
		t.Fatalf("expected NoImageGeneratedError, got %T %v", err, err)
	}
	var noImage *NoImageGeneratedError
	if !errors.As(err, &noImage) || len(noImage.Responses) != 1 || noImage.Responses[0].ID != "response-1" {
		t.Fatalf("expected response metadata on error, got %#v", err)
	}
}

func TestGenerateSpeechAndTranscribeUseNamedErrors(t *testing.T) {
	speech := NewMockSpeechModel("speech")
	speech.GenerateFunc = func(_ context.Context, _ SpeechModelCallOptions) (*SpeechModelResult, error) {
		return &SpeechModelResult{Response: ResponseMetadata{ID: "speech-response"}}, nil
	}
	if _, err := GenerateSpeech(context.Background(), GenerateSpeechOptions{Model: speech, Text: "hello"}); !IsNoSpeechGeneratedError(err) {
		t.Fatalf("expected NoSpeechGeneratedError, got %T %v", err, err)
	}

	transcription := NewMockTranscriptionModel("transcription")
	transcription.TranscribeFunc = func(_ context.Context, _ TranscriptionModelCallOptions) (*TranscriptionModelResult, error) {
		return &TranscriptionModelResult{Response: ResponseMetadata{ID: "transcribe-response"}}, nil
	}
	if _, err := Transcribe(context.Background(), TranscribeOptions{Model: transcription}); !IsNoTranscriptGeneratedError(err) {
		t.Fatalf("expected NoTranscriptGeneratedError, got %T %v", err, err)
	}
}

func TestGenerateVideoNoVideosUsesNamedError(t *testing.T) {
	model := NewMockVideoModel("video")
	model.GenerateFunc = func(_ context.Context, _ VideoModelCallOptions) (*VideoModelResult, error) {
		return &VideoModelResult{Response: ResponseMetadata{ID: "response-1"}}, nil
	}
	_, err := GenerateVideo(context.Background(), GenerateVideoOptions{Model: model, Prompt: "make"})
	if !IsNoVideoGeneratedError(err) {
		t.Fatalf("expected NoVideoGeneratedError, got %T %v", err, err)
	}
}

func TestRerankBuildsDocumentViewsAndEmitsCallbacks(t *testing.T) {
	model := NewMockRerankingModel("rerank")
	model.RerankFunc = func(_ context.Context, opts RerankingModelCallOptions) (*RerankingModelResult, error) {
		if !reflect.DeepEqual(opts.Documents, []string{"a", "b", "c"}) {
			t.Fatalf("unexpected documents: %#v", opts.Documents)
		}
		return &RerankingModelResult{
			Results: []RerankingResult{
				{Index: 2, Score: 0.9},
				{Index: 0, Score: 0.8},
			},
			Warnings: []Warning{{Type: "other", Message: "warning"}},
		}, nil
	}
	var calls []string
	result, err := Rerank(context.Background(), RerankOptions{
		Model:     model,
		Query:     "q",
		Documents: []string{"a", "b", "c"},
		OnStart: func(event StartEvent) {
			calls = append(calls, event.Operation)
		},
		OnFinish: func(event FinishEvent) {
			calls = append(calls, event.Operation)
		},
	})
	if err != nil {
		t.Fatalf("Rerank failed: %v", err)
	}
	if !reflect.DeepEqual(result.OriginalDocuments, []string{"a", "b", "c"}) {
		t.Fatalf("unexpected original documents: %#v", result.OriginalDocuments)
	}
	if !reflect.DeepEqual(result.RerankedDocuments, []string{"c", "a"}) {
		t.Fatalf("unexpected reranked documents: %#v", result.RerankedDocuments)
	}
	if result.Ranking[0].Document != "c" || result.Ranking[1].Document != "a" {
		t.Fatalf("expected documents to be filled into ranking: %#v", result.Ranking)
	}
	if !reflect.DeepEqual(calls, []string{"rerank", "rerank"}) {
		t.Fatalf("unexpected callback calls: %#v", calls)
	}
}

func TestRerankRejectsOutOfRangeIndex(t *testing.T) {
	model := NewMockRerankingModel("rerank")
	model.RerankFunc = func(_ context.Context, _ RerankingModelCallOptions) (*RerankingModelResult, error) {
		return &RerankingModelResult{Results: []RerankingResult{{Index: 3, Score: 1}}}, nil
	}
	_, err := Rerank(context.Background(), RerankOptions{Model: model, Query: "q", Documents: []string{"a"}})
	if !errors.Is(err, ErrInvalidResponseData) {
		t.Fatalf("expected invalid response data, got %T %v", err, err)
	}
}
