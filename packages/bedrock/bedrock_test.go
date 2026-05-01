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
