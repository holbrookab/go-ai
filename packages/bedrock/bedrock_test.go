package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/holbrookab/go-ai/packages/ai"
)

func TestDoGenerateUsesAPIKeyAndMapsRequest(t *testing.T) {
	client := &captureClient{response: `{
		"output":{"message":{"content":[{"text":"hello"}]}},
		"stopReason":"end_turn",
		"usage":{"inputTokens":2,"outputTokens":3}
	}`}
	provider := New(Settings{
		Region:     "us-east-1",
		APIKey:     "token",
		Client:     client,
		GenerateID: func() string { return "id" },
	})
	model := provider.LanguageModel("anthropic.claude-3-haiku-20240307-v1:0")
	result, err := model.DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt:          []ai.Message{ai.SystemMessage("system"), ai.UserMessage("hello")},
		MaxOutputTokens: ptr(100),
		Temperature:     fptr(2),
		Tools: []ai.ModelTool{{
			Type:        "function",
			Name:        "weather",
			Description: "Weather",
			InputSchema: map[string]any{"type": "object"},
		}},
		ToolChoice: ai.RequiredToolChoice(),
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if got := client.request.Header.Get("Authorization"); got != "Bearer token" {
		t.Fatalf("expected bearer token, got %q", got)
	}
	if !strings.Contains(client.request.URL.Path, "/model/anthropic.claude-3-haiku-20240307-v1:0/converse") {
		t.Fatalf("unexpected path: %s", client.request.URL.Path)
	}
	var body map[string]any
	if err := json.Unmarshal(client.body, &body); err != nil {
		t.Fatal(err)
	}
	inference := body["inferenceConfig"].(map[string]any)
	if inference["temperature"].(float64) != 1 {
		t.Fatalf("expected temperature clamped to 1, got %#v", inference["temperature"])
	}
	if _, ok := body["toolConfig"]; !ok {
		t.Fatalf("expected toolConfig in request: %#v", body)
	}
	if ai.TextFromParts(result.Content) != "hello" {
		t.Fatalf("expected parsed text, got %q", ai.TextFromParts(result.Content))
	}
	if len(result.Warnings) == 0 {
		t.Fatalf("expected temperature warning")
	}
}

func TestDoGenerateFallsBackToSigV4AndDoesNotUseEnvSessionWhenExplicitKeys(t *testing.T) {
	t.Setenv("AWS_SESSION_TOKEN", "env-session")
	client := &captureClient{response: `{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}`}
	provider := New(Settings{
		Region:          "us-east-1",
		AccessKeyID:     "AKID",
		SecretAccessKey: "SECRET",
		SessionToken:    "explicit-session",
		Client:          client,
	})
	_, err := provider.LanguageModel("amazon.nova-lite-v1:0").DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if got := client.request.Header.Get("Authorization"); !strings.HasPrefix(got, "AWS4-HMAC-SHA256") {
		t.Fatalf("expected sigv4 authorization, got %q", got)
	}
	if got := client.request.Header.Get("X-Amz-Security-Token"); got != "explicit-session" {
		t.Fatalf("expected explicit session token, got %q", got)
	}
}

func TestBlankAPIKeyFallsBackToSigV4(t *testing.T) {
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "  ")
	client := &captureClient{response: `{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}`}
	provider := New(Settings{
		Region:          "us-east-1",
		APIKey:          " ",
		AccessKeyID:     "AKID",
		SecretAccessKey: "SECRET",
		Client:          client,
	})
	_, err := provider.LanguageModel("amazon.nova-lite-v1:0").DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if got := client.request.Header.Get("Authorization"); !strings.HasPrefix(got, "AWS4-HMAC-SHA256") {
		t.Fatalf("expected sigv4 authorization, got %q", got)
	}
}

func TestDoGenerateParsesUpstreamToolFixture(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "amazon-bedrock-tool-call.1.json"))
	if err != nil {
		t.Fatal(err)
	}
	client := &captureClient{response: string(data)}
	provider := New(Settings{
		Region:  "us-east-1",
		APIKey:  "token",
		Client:  client,
		Headers: map[string]string{"X-Test": "true"},
	})
	result, err := provider.LanguageModel("anthropic.claude-3-haiku-20240307-v1:0").DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("run ls")},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if result.FinishReason.Unified != ai.FinishToolCalls {
		t.Fatalf("expected tool-call finish, got %#v", result.FinishReason)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected one content part, got %#v", result.Content)
	}
	call, ok := result.Content[0].(ai.ToolCallPart)
	if !ok {
		t.Fatalf("expected tool call, got %#v", result.Content[0])
	}
	if call.ToolCallID != "tool-use-id" || call.ToolName != "bash" || call.InputRaw != `{"command":"ls -l"}` {
		t.Fatalf("unexpected tool call: %#v", call)
	}
}

func TestDoGenerateSkipsEmptyTextBeforeJSONToolResponse(t *testing.T) {
	client := &captureClient{response: `{
		"output":{"message":{"content":[
			{"text":""},
			{"toolUse":{"toolUseId":"json-id","name":"json","input":{"skills":[],"execution":"parallel","reason":"Greeting only"}}}
		]}},
		"stopReason":"tool_use"
	}`}
	provider := New(Settings{
		Region: "us-east-1",
		APIKey: "token",
		Client: client,
	})
	result, err := provider.LanguageModel("amazon.nova-lite-v1:0").DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("Aloha")},
		ResponseFormat: &ai.ResponseFormat{
			Type:   "json",
			Schema: map[string]any{"type": "object"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected only the JSON text part, got %#v", result.Content)
	}
	if got := ai.TextFromParts(result.Content); got != `{"execution":"parallel","reason":"Greeting only","skills":[]}` {
		t.Fatalf("unexpected JSON response text: %q", got)
	}
}

func TestDoGenerateMapsMessageTextWhenContentIsEmpty(t *testing.T) {
	client := &captureClient{response: `{
		"output":{"message":{"content":[{"text":"aloha"}]}},
		"stopReason":"end_turn"
	}`}
	provider := New(Settings{
		Region: "us-east-1",
		APIKey: "token",
		Client: client,
	})
	_, err := provider.LanguageModel("amazon.nova-lite-v1:0").DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{
			{Role: ai.RoleUser, Text: "hello from text field"},
			{Role: ai.RoleAssistant, Text: "assistant from text field"},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	var body struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(client.body, &body); err != nil {
		t.Fatal(err)
	}
	if got := body.Messages[0].Content[0].Text; got != "hello from text field" {
		t.Fatalf("expected user text fallback, got %q", got)
	}
	if got := body.Messages[1].Content[0].Text; got != "assistant from text field" {
		t.Fatalf("expected assistant text fallback, got %q", got)
	}
}

func TestDoStreamDecodesBedrockEventStreamFrames(t *testing.T) {
	var stream bytes.Buffer
	encoder := eventstream.NewEncoder()
	for _, payload := range []string{
		`{"contentBlockStart":{"contentBlockIndex":0,"start":{}}}`,
		`{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"hello"}}}`,
		`{"contentBlockStop":{"contentBlockIndex":0}}`,
		`{"messageStop":{"stopReason":"end_turn"}}`,
		`{"metadata":{"usage":{"inputTokens":2,"outputTokens":3,"totalTokens":5}}}`,
	} {
		if err := encoder.Encode(&stream, eventstream.Message{Payload: []byte(payload)}); err != nil {
			t.Fatalf("encode event stream frame: %v", err)
		}
	}
	client := &captureClient{response: stream.String()}
	provider := New(Settings{
		Region: "us-east-1",
		APIKey: "token",
		Client: client,
	})
	result, err := provider.LanguageModel("amazon.nova-lite-v1:0").DoStream(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("DoStream failed: %v", err)
	}
	var text string
	var finish ai.StreamPart
	for part := range result.Stream {
		if part.Type == "text-delta" {
			text += part.TextDelta
		}
		if part.Type == "finish" {
			finish = part
		}
		if part.Type == "error" {
			t.Fatalf("unexpected stream error: %v", part.Err)
		}
	}
	if text != "hello" {
		t.Fatalf("expected streamed text, got %q", text)
	}
	if finish.FinishReason.Unified != ai.FinishStop || intValue(finish.Usage.InputTokens) != 2 || intValue(finish.Usage.OutputTokens) != 3 || intValue(finish.Usage.TotalTokens) != 5 {
		t.Fatalf("unexpected finish part: %#v", finish)
	}
}

func TestDoStreamDecodesBedrockEventStreamHeaders(t *testing.T) {
	var stream bytes.Buffer
	encoder := eventstream.NewEncoder()
	events := []struct {
		eventType string
		payload   string
	}{
		{"messageStart", `{"role":"assistant"}`},
		{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"Hello!"},"p":"opaque"}`},
		{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":" How can I help?"}}`},
		{"contentBlockStop", `{"contentBlockIndex":0}`},
		{"messageStop", `{"stopReason":"end_turn"}`},
		{"metadata", `{"usage":{"inputTokens":2,"outputTokens":3,"totalTokens":5},"metrics":{"latencyMs":1}}`},
	}
	for _, event := range events {
		if err := encoder.Encode(&stream, eventstream.Message{
			Headers: eventstream.Headers{
				{Name: ":event-type", Value: eventstream.StringValue(event.eventType)},
			},
			Payload: []byte(event.payload),
		}); err != nil {
			t.Fatalf("encode event stream frame: %v", err)
		}
	}
	client := &captureClient{response: stream.String()}
	provider := New(Settings{
		Region: "us-east-1",
		APIKey: "token",
		Client: client,
	})
	result, err := provider.LanguageModel("amazon.nova-lite-v1:0").DoStream(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("DoStream failed: %v", err)
	}
	var text string
	var finish ai.StreamPart
	var textDeltaCount int
	for part := range result.Stream {
		if part.Type == "text-delta" {
			text += part.TextDelta
			textDeltaCount++
		}
		if part.Type == "finish" {
			finish = part
		}
		if part.Type == "error" {
			t.Fatalf("unexpected stream error: %v", part.Err)
		}
	}
	if text != "Hello! How can I help?" || textDeltaCount != 2 {
		t.Fatalf("expected classified text deltas, got text %q count %d", text, textDeltaCount)
	}
	if finish.FinishReason.Unified != ai.FinishStop || finish.FinishReason.Raw != "end_turn" || intValue(finish.Usage.TotalTokens) != 5 {
		t.Fatalf("unexpected finish part: %#v", finish)
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}

type captureClient struct {
	request  *http.Request
	body     []byte
	response string
}

func (c *captureClient) Do(req *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(req.Body)
	req.Body = io.NopCloser(bytes.NewReader(body))
	c.request = req
	c.body = body
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(c.response)),
	}, nil
}

func ptr(v int) *int          { return &v }
func fptr(v float64) *float64 { return &v }

func intValue(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}
