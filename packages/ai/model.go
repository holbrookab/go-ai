package ai

import (
	"fmt"
	"reflect"
)

type LanguageModelResolver interface {
	LanguageModel(ref string) (LanguageModel, error)
}

type EmbeddingModelResolver interface {
	EmbeddingModel(ref string) (EmbeddingModel, error)
}

type ImageModelResolver interface {
	ImageModel(ref string) (ImageModel, error)
}

type VideoModelResolver interface {
	VideoModel(ref string) (VideoModel, error)
}

type SpeechModelResolver interface {
	SpeechModel(ref string) (SpeechModel, error)
}

type TranscriptionModelResolver interface {
	TranscriptionModel(ref string) (TranscriptionModel, error)
}

type RerankingModelResolver interface {
	RerankingModel(ref string) (RerankingModel, error)
}

func ResolveLanguageModel(model any, resolver LanguageModelResolver) (LanguageModel, error) {
	switch model := model.(type) {
	case LanguageModel:
		if isNil(model) {
			return nil, noSuchModelError("language model", "")
		}
		return model, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "language model resolver is required"}
		}
		resolved, err := resolver.LanguageModel(model)
		if err != nil {
			return nil, err
		}
		if isNil(resolved) {
			return nil, noSuchModelError("language model", model)
		}
		return resolved, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "language model is required"}
	default:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("unsupported language model reference %T", model)}
	}
}

func ResolveEmbeddingModel(model any, resolver EmbeddingModelResolver) (EmbeddingModel, error) {
	switch model := model.(type) {
	case EmbeddingModel:
		if isNil(model) {
			return nil, noSuchModelError("embedding model", "")
		}
		return model, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "embedding model resolver is required"}
		}
		resolved, err := resolver.EmbeddingModel(model)
		if err != nil {
			return nil, err
		}
		if isNil(resolved) {
			return nil, noSuchModelError("embedding model", model)
		}
		return resolved, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "embedding model is required"}
	default:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("unsupported embedding model reference %T", model)}
	}
}

func ResolveImageModel(model any, resolver ImageModelResolver) (ImageModel, error) {
	switch model := model.(type) {
	case ImageModel:
		if isNil(model) {
			return nil, noSuchModelError("image model", "")
		}
		return model, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "image model resolver is required"}
		}
		resolved, err := resolver.ImageModel(model)
		if err != nil {
			return nil, err
		}
		if isNil(resolved) {
			return nil, noSuchModelError("image model", model)
		}
		return resolved, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "image model is required"}
	default:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("unsupported image model reference %T", model)}
	}
}

func ResolveVideoModel(model any, resolver VideoModelResolver) (VideoModel, error) {
	switch model := model.(type) {
	case VideoModel:
		if isNil(model) {
			return nil, noSuchModelError("video model", "")
		}
		return model, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "video model resolver is required"}
		}
		resolved, err := resolver.VideoModel(model)
		if err != nil {
			return nil, err
		}
		if isNil(resolved) {
			return nil, noSuchModelError("video model", model)
		}
		return resolved, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "video model is required"}
	default:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("unsupported video model reference %T", model)}
	}
}

func ResolveSpeechModel(model any, resolver SpeechModelResolver) (SpeechModel, error) {
	switch model := model.(type) {
	case SpeechModel:
		if isNil(model) {
			return nil, noSuchModelError("speech model", "")
		}
		return model, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "speech model resolver is required"}
		}
		resolved, err := resolver.SpeechModel(model)
		if err != nil {
			return nil, err
		}
		if isNil(resolved) {
			return nil, noSuchModelError("speech model", model)
		}
		return resolved, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "speech model is required"}
	default:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("unsupported speech model reference %T", model)}
	}
}

func ResolveTranscriptionModel(model any, resolver TranscriptionModelResolver) (TranscriptionModel, error) {
	switch model := model.(type) {
	case TranscriptionModel:
		if isNil(model) {
			return nil, noSuchModelError("transcription model", "")
		}
		return model, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "transcription model resolver is required"}
		}
		resolved, err := resolver.TranscriptionModel(model)
		if err != nil {
			return nil, err
		}
		if isNil(resolved) {
			return nil, noSuchModelError("transcription model", model)
		}
		return resolved, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "transcription model is required"}
	default:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("unsupported transcription model reference %T", model)}
	}
}

func ResolveRerankingModel(model any, resolver RerankingModelResolver) (RerankingModel, error) {
	switch model := model.(type) {
	case RerankingModel:
		if isNil(model) {
			return nil, noSuchModelError("reranking model", "")
		}
		return model, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "reranking model resolver is required"}
		}
		resolved, err := resolver.RerankingModel(model)
		if err != nil {
			return nil, err
		}
		if isNil(resolved) {
			return nil, noSuchModelError("reranking model", model)
		}
		return resolved, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "reranking model is required"}
	default:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("unsupported reranking model reference %T", model)}
	}
}

func noSuchModelError(modelType, ref string) *SDKError {
	if ref == "" {
		return &SDKError{Kind: ErrNoSuchModel, Message: modelType}
	}
	return &SDKError{Kind: ErrNoSuchModel, Message: fmt.Sprintf("%s %q", modelType, ref)}
}

func isNil(v any) bool {
	if v == nil {
		return true
	}
	value := reflect.ValueOf(v)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
