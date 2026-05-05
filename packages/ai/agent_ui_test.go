package ai

import (
	"context"
	"testing"
)

func TestCreateAgentUIStreamMapsTextAndFinishChunks(t *testing.T) {
	model := NewMockLanguageModel("stream")
	model.StreamFunc = func(context.Context, LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		stream := make(chan StreamPart, 2)
		stream <- StreamPart{Type: "text-delta", TextDelta: "hello"}
		stream <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(stream)
		return &LanguageModelStreamResult{Stream: stream}, nil
	}
	agent := NewToolLoopAgent(ToolLoopAgentSettings{Model: model})

	chunks, err := ReadUIMessageStream(CreateAgentUIStream(context.Background(), AgentUIStreamOptions{
		Agent:     agent,
		Call:      AgentStreamOptions{AgentCallOptions: AgentCallOptions{Prompt: "hi"}},
		MessageID: "message-1",
		TextID:    "text-1",
	}))
	if err != nil {
		t.Fatal(err)
	}
	gotTypes := make([]string, len(chunks))
	for i, chunk := range chunks {
		gotTypes[i] = chunk.Type
	}
	want := []string{
		UIMessageChunkTypeStart,
		UIMessageChunkTypeStartStep,
		UIMessageChunkTypeTextStart,
		UIMessageChunkTypeTextDelta,
		UIMessageChunkTypeFinishStep,
		UIMessageChunkTypeTextEnd,
		UIMessageChunkTypeFinish,
	}
	if len(gotTypes) != len(want) {
		t.Fatalf("types = %#v, want %#v", gotTypes, want)
	}
	for i := range want {
		if gotTypes[i] != want[i] {
			t.Fatalf("types = %#v, want %#v", gotTypes, want)
		}
	}
	if chunks[1].StepID != "step-0" || chunks[1].StepNumber == nil || *chunks[1].StepNumber != 0 || chunks[1].StepType != "initial" {
		t.Fatalf("missing step metadata: %#v", chunks[1])
	}
	if chunks[3].Delta != "hello" || chunks[len(chunks)-1].FinishReason != FinishStop {
		t.Fatalf("unexpected chunks: %#v", chunks)
	}
}

func TestCreateAgentUIStreamMapsToolChunks(t *testing.T) {
	model := NewMockLanguageModel("stream")
	model.StreamFunc = func(context.Context, LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		stream := make(chan StreamPart, 4)
		stream <- StreamPart{Type: "tool-call", ToolCallID: "call-1", ToolName: "weather", ToolInput: `{"city":"NYC"}`}
		stream <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishToolCalls}}
		close(stream)
		return &LanguageModelStreamResult{Stream: stream}, nil
	}
	agent := NewToolLoopAgent(ToolLoopAgentSettings{
		Model: model,
		Tools: map[string]Tool{
			"weather": {
				Execute: func(context.Context, ToolCall, ToolExecutionOptions) (any, error) {
					return "sunny", nil
				},
			},
		},
		StopWhen: []StopCondition{StepCount(1)},
	})

	chunks, err := ReadUIMessageStream(CreateAgentUIStream(context.Background(), AgentUIStreamOptions{
		Agent: agent,
		Call:  AgentStreamOptions{AgentCallOptions: AgentCallOptions{Prompt: "weather"}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	var sawInput, sawOutput bool
	for _, chunk := range chunks {
		if chunk.Type == UIMessageChunkTypeToolInputAvailable && chunk.ToolName == "weather" {
			sawInput = true
		}
		if chunk.Type == UIMessageChunkTypeToolOutputAvailable && chunk.ToolCallID == "call-1" && chunk.Output == "sunny" {
			sawOutput = true
		}
	}
	if !sawInput || !sawOutput {
		t.Fatalf("expected tool input/output chunks, got %#v", chunks)
	}
}

func TestCreateStreamTextUIMessageStreamMapsAbortChunk(t *testing.T) {
	stream := make(chan StreamPart, 1)
	stream <- StreamPart{Type: "abort", AbortReason: "manual abort"}
	close(stream)

	chunks, err := ReadUIMessageStream(CreateStreamTextUIMessageStream(context.Background(), &StreamTextResult{Stream: stream}))
	if err != nil {
		t.Fatal(err)
	}
	var sawAbort bool
	for _, chunk := range chunks {
		if chunk.Type == UIMessageChunkTypeAbort {
			sawAbort = true
			if chunk.Reason != "manual abort" {
				t.Fatalf("abort reason = %q", chunk.Reason)
			}
		}
	}
	if !sawAbort {
		t.Fatalf("expected abort chunk, got %#v", chunks)
	}
}
