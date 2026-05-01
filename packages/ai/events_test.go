package ai

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
)

func TestGenerateTextLifecycleEvents(t *testing.T) {
	telemetry := &recordingTelemetry{}
	var callbacks []string
	result, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:     mockModel{},
		Prompt:    "hello",
		Telemetry: telemetry,
		OnStart: func(event StartEvent) {
			callbacks = append(callbacks, event.Operation+":start")
		},
		OnStepFinish: func(event StepFinishEvent) {
			if event.Step == nil || event.Step.Text != "ok" {
				t.Fatalf("unexpected step event: %#v", event)
			}
			callbacks = append(callbacks, event.Operation+":step")
		},
		OnFinish: func(event FinishEvent) {
			if event.Result == nil {
				t.Fatalf("finish event missing result")
			}
			callbacks = append(callbacks, event.Operation+":finish")
		},
	})
	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}
	if result.Text != "ok" {
		t.Fatalf("unexpected text: %q", result.Text)
	}
	if want := []string{"generate_text:start", "generate_text:step", "generate_text:finish"}; !reflect.DeepEqual(callbacks, want) {
		t.Fatalf("callbacks = %#v, want %#v", callbacks, want)
	}
	if want := []string{EventGenerateTextStart, EventOnLanguageModelCallStart, EventOnLanguageModelCallEnd, EventGenerateTextStepFinish, EventGenerateTextFinish}; !reflect.DeepEqual(telemetry.names(), want) {
		t.Fatalf("telemetry = %#v, want %#v", telemetry.names(), want)
	}
	events := telemetry.snapshot()
	if events[0].OperationID != OperationIDGenerateText || events[1].Name != EventOnLanguageModelCallStart || events[1].CallID == "" {
		t.Fatalf("unexpected diagnostic telemetry events: %#v", events)
	}
}

func TestStreamTextLifecycleChunkAndErrorEvents(t *testing.T) {
	telemetry := &recordingTelemetry{}
	var chunks []string
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 2)
		ch <- StreamPart{Type: "text-delta", TextDelta: "hi"}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}
	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:     model,
			Prompt:    "hello",
			Telemetry: telemetry,
		},
		OnChunk: func(event ChunkEvent) {
			chunks = append(chunks, event.Chunk.Type)
		},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}
	for range result.Stream {
	}
	if want := []string{"text-delta", "finish-step", "finish"}; !reflect.DeepEqual(chunks, want) {
		t.Fatalf("chunks = %#v, want %#v", chunks, want)
	}
	if !containsEvent(telemetry.names(), EventStreamTextFinish) {
		t.Fatalf("expected stream finish telemetry, got %#v", telemetry.names())
	}

	boom := errors.New("boom")
	var errorEvent error
	_, err = GenerateText(context.Background(), GenerateTextOptions{
		Model:  &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) { return nil, boom }},
		Prompt: "hello",
		OnError: func(event ErrorEvent) {
			errorEvent = event.Err
		},
	})
	if !errors.Is(err, boom) {
		t.Fatalf("expected boom, got %v", err)
	}
	if !errors.Is(errorEvent, boom) {
		t.Fatalf("expected error callback to receive boom, got %v", errorEvent)
	}
}

func TestEmbedManyAndGenerateObjectEvents(t *testing.T) {
	embedTelemetry := &recordingTelemetry{}
	_, err := EmbedMany(context.Background(), EmbedManyOptions{
		Model:     &mockEmbeddingModel{max: 2},
		Values:    []string{"a", "b"},
		Telemetry: embedTelemetry,
	})
	if err != nil {
		t.Fatalf("EmbedMany failed: %v", err)
	}
	if want := []string{EventEmbedManyStart, EventEmbedManyFinish}; !reflect.DeepEqual(embedTelemetry.names(), want) {
		t.Fatalf("embed telemetry = %#v, want %#v", embedTelemetry.names(), want)
	}

	objectTelemetry := &recordingTelemetry{}
	_, err = GenerateObject(context.Background(), GenerateObjectOptions{
		Model:     mockModel{},
		Prompt:    "json",
		Schema:    map[string]any{"type": "object"},
		Telemetry: objectTelemetry,
	})
	if err == nil {
		t.Fatalf("expected JSON parse error")
	}
	if want := []string{EventGenerateObjectStart, EventGenerateObjectError}; !reflect.DeepEqual(objectTelemetry.names(), want) {
		t.Fatalf("object telemetry = %#v, want %#v", objectTelemetry.names(), want)
	}
}

func TestTelemetryOptionsFilterAttributes(t *testing.T) {
	recordInputs := false
	recordOutputs := false
	telemetry := &recordingTelemetry{}
	_, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:     mockModel{},
		Prompt:    "hello",
		Telemetry: telemetry,
		TelemetryOptions: TelemetryOptions{
			RecordInputs:  &recordInputs,
			RecordOutputs: &recordOutputs,
			FunctionID:    "fn",
			AttributeFilter: func(_ Event, key string, _ any) bool {
				return key != "usage"
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range telemetry.snapshot() {
		if event.FunctionID != "fn" || event.RecordInputs == nil || *event.RecordInputs || event.RecordOutputs == nil || *event.RecordOutputs {
			t.Fatalf("telemetry options not applied to event: %#v", event)
		}
		if _, ok := event.Attributes["input.prompt"]; ok {
			t.Fatalf("input attribute should be filtered: %#v", event.Attributes)
		}
		if _, ok := event.Attributes["output.content"]; ok {
			t.Fatalf("output attribute should be filtered: %#v", event.Attributes)
		}
		if _, ok := event.Attributes["usage"]; ok {
			t.Fatalf("custom attribute filter should remove usage: %#v", event.Attributes)
		}
	}
}

func TestTelemetryDisabledSuppressesRecordEvent(t *testing.T) {
	enabled := false
	telemetry := &recordingTelemetry{}
	_, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:     mockModel{},
		Prompt:    "hello",
		Telemetry: telemetry,
		TelemetryOptions: TelemetryOptions{
			IsEnabled: &enabled,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := telemetry.names(); len(got) != 0 {
		t.Fatalf("expected no telemetry when disabled, got %#v", got)
	}
}

type recordingTelemetry struct {
	mu     sync.Mutex
	events []Event
}

func (r *recordingTelemetry) RecordEvent(_ context.Context, event Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, event)
}

func (r *recordingTelemetry) names() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	names := make([]string, len(r.events))
	for i, event := range r.events {
		names[i] = event.Name
	}
	return names
}

func (r *recordingTelemetry) snapshot() []Event {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Event(nil), r.events...)
}

func containsEvent(names []string, want string) bool {
	for _, name := range names {
		if name == want {
			return true
		}
	}
	return false
}
