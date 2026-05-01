package ai

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestInvalidStreamPartError(t *testing.T) {
	chunk := StreamPart{Type: "text-delta", TextDelta: "hi"}
	err := NewInvalidStreamPartError(chunk, "bad stream part")

	if !errors.Is(err, ErrInvalidStreamPart) {
		t.Fatalf("expected invalid stream part kind, got %v", err)
	}
	if !IsInvalidStreamPartError(err) {
		t.Fatalf("expected typed invalid stream part detection")
	}
	var typed *InvalidStreamPartError
	if !errors.As(fmt.Errorf("wrapped: %w", err), &typed) {
		t.Fatalf("expected errors.As to find InvalidStreamPartError")
	}
	if typed.Chunk.Type != "text-delta" || typed.Chunk.TextDelta != "hi" {
		t.Fatalf("unexpected chunk: %#v", typed.Chunk)
	}
}

func TestInvalidDataContentError(t *testing.T) {
	err := NewInvalidDataContentError(42, errors.New("bad data"), "")

	if !errors.Is(err, ErrInvalidDataContent) {
		t.Fatalf("expected invalid data content kind, got %v", err)
	}
	if !IsInvalidDataContentError(fmt.Errorf("wrapped: %w", err)) {
		t.Fatalf("expected typed invalid data content detection")
	}
	if err.Content != 42 {
		t.Fatalf("unexpected content: %#v", err.Content)
	}
}

func TestInvalidMessageRoleError(t *testing.T) {
	err := NewInvalidMessageRoleError(Role("developer"))

	if !errors.Is(err, ErrInvalidMessageRole) {
		t.Fatalf("expected invalid message role kind, got %v", err)
	}
	if !IsInvalidMessageRoleError(fmt.Errorf("wrapped: %w", err)) {
		t.Fatalf("expected typed invalid message role detection")
	}
	if err.Role != Role("developer") {
		t.Fatalf("unexpected role: %q", err.Role)
	}
}

func TestMessageConversionError(t *testing.T) {
	message := Message{Role: RoleUser, Text: "hello"}
	err := NewMessageConversionError(message, "cannot convert message")

	if !errors.Is(err, ErrMessageConversion) {
		t.Fatalf("expected message conversion kind, got %v", err)
	}
	if !IsMessageConversionError(fmt.Errorf("wrapped: %w", err)) {
		t.Fatalf("expected typed message conversion detection")
	}
	if !reflect.DeepEqual(err.OriginalMessage, message) {
		t.Fatalf("unexpected original message: %#v", err.OriginalMessage)
	}
}

func TestDownloadError(t *testing.T) {
	cause := errors.New("network")
	err := NewDownloadError("https://example.com/file", 0, "", "", cause)

	if !errors.Is(err, ErrDownload) {
		t.Fatalf("expected download error kind, got %v", err)
	}
	if !IsDownloadError(fmt.Errorf("wrapped: %w", err)) {
		t.Fatalf("expected typed download detection")
	}
	if err.URL != "https://example.com/file" || err.StatusCode != 0 || err.StatusText != "" {
		t.Fatalf("unexpected download fields: %#v", err)
	}
}

func TestWrapGatewayError(t *testing.T) {
	authErr := NewGatewayAuthenticationError("", nil)
	wrapped := WrapGatewayError(authErr)
	if !errors.Is(wrapped, ErrGateway) || !IsGatewayError(wrapped) {
		t.Fatalf("expected gateway wrapper, got %v", wrapped)
	}
	if WrapGatewayError(errors.New("other")).Error() != "other" {
		t.Fatalf("non-gateway errors should pass through")
	}
}

func TestInvalidToolApprovalError(t *testing.T) {
	err := NewInvalidToolApprovalError("approval-1")

	if !errors.Is(err, ErrInvalidToolApproval) {
		t.Fatalf("expected invalid tool approval kind, got %v", err)
	}
	if !IsInvalidToolApprovalError(err) {
		t.Fatalf("expected typed invalid tool approval detection")
	}
	if err.ApprovalID != "approval-1" {
		t.Fatalf("unexpected approval id: %q", err.ApprovalID)
	}
	want := `invalid tool approval: Tool approval response references unknown approvalId: "approval-1". No matching tool-approval-request found in message history.`
	if err.Error() != want {
		t.Fatalf("unexpected message:\nwant %q\n got %q", want, err.Error())
	}
}

func TestToolCallNotFoundForApprovalError(t *testing.T) {
	err := NewToolCallNotFoundForApprovalError("call-1", "approval-1")

	if !errors.Is(err, ErrToolCallNotFoundForApproval) {
		t.Fatalf("expected tool-call-not-found kind, got %v", err)
	}
	if !IsToolCallNotFoundForApprovalError(err) {
		t.Fatalf("expected typed tool-call-not-found detection")
	}
	if err.ToolCallID != "call-1" || err.ApprovalID != "approval-1" {
		t.Fatalf("unexpected ids: %#v", err)
	}
}

func TestToolCallRepairError(t *testing.T) {
	original := NewNoSuchToolError("weather", []string{"time"})
	cause := errors.New("repair failed")
	err := NewToolCallRepairError(cause, original)

	if !errors.Is(err, ErrToolCallRepair) {
		t.Fatalf("expected tool call repair kind, got %v", err)
	}
	if !IsToolCallRepairError(fmt.Errorf("wrapped: %w", err)) {
		t.Fatalf("expected wrapped typed tool call repair detection")
	}
	if err.OriginalError != original {
		t.Fatalf("unexpected original error: %#v", err.OriginalError)
	}
	want := "tool call repair error: Error repairing tool call: repair failed: repair failed"
	if err.Error() != want {
		t.Fatalf("unexpected message:\nwant %q\n got %q", want, err.Error())
	}
}

func TestUnsupportedModelVersionError(t *testing.T) {
	err := NewUnsupportedModelVersionError("v1", "mock", "old-model")

	if !errors.Is(err, ErrUnsupportedModelVersion) {
		t.Fatalf("expected unsupported model version kind, got %v", err)
	}
	if !IsUnsupportedModelVersionError(err) {
		t.Fatalf("expected typed unsupported model version detection")
	}
	if err.Version != "v1" || err.Provider != "mock" || err.ModelID != "old-model" {
		t.Fatalf("unexpected fields: %#v", err)
	}
}

func TestNoObjectGeneratedError(t *testing.T) {
	err := NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
		Text:         "{",
		Response:     ResponseMetadata{ID: "response-1"},
		Usage:        usage(1, 2),
		FinishReason: FinishReason{Unified: FinishStop},
	})

	if !errors.Is(err, ErrNoObjectGenerated) {
		t.Fatalf("expected no object generated kind, got %v", err)
	}
	if !IsNoObjectGeneratedError(err) {
		t.Fatalf("expected typed no object generated detection")
	}
	if err.Text != "{" || err.Response.ID != "response-1" || err.FinishReason.Unified != FinishStop {
		t.Fatalf("unexpected fields: %#v", err)
	}
}

func TestNoMediaGeneratedErrors(t *testing.T) {
	responses := []ResponseMetadata{{ID: "response-1"}}
	imageErr := NewNoImageGeneratedError(NoImageGeneratedErrorOptions{Responses: responses})
	speechErr := NewNoSpeechGeneratedError(responses)
	transcriptErr := NewNoTranscriptGeneratedError(responses)
	videoErr := NewNoVideoGeneratedError(NoVideoGeneratedErrorOptions{Responses: responses})

	cases := []struct {
		name string
		err  error
		kind error
		is   func(error) bool
	}{
		{name: "image", err: imageErr, kind: ErrNoImageGenerated, is: IsNoImageGeneratedError},
		{name: "speech", err: speechErr, kind: ErrNoSpeechGenerated, is: IsNoSpeechGeneratedError},
		{name: "transcript", err: transcriptErr, kind: ErrNoTranscriptGenerated, is: IsNoTranscriptGeneratedError},
		{name: "video", err: videoErr, kind: ErrNoVideoGenerated, is: IsNoVideoGeneratedError},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !errors.Is(tc.err, tc.kind) {
				t.Fatalf("expected kind %v, got %v", tc.kind, tc.err)
			}
			if !tc.is(fmt.Errorf("wrapped: %w", tc.err)) {
				t.Fatalf("expected typed detection")
			}
		})
	}
	if imageErr.Responses[0].ID != "response-1" || speechErr.Responses[0].ID != "response-1" ||
		transcriptErr.Responses[0].ID != "response-1" || videoErr.Responses[0].ID != "response-1" {
		t.Fatalf("responses were not preserved")
	}
}

func TestUIMessageStreamError(t *testing.T) {
	err := NewUIMessageStreamError("text-delta", "part-1", "delta without start")

	if !errors.Is(err, ErrUIMessageStream) {
		t.Fatalf("expected ui message stream kind, got %v", err)
	}
	if !IsUIMessageStreamError(err) {
		t.Fatalf("expected typed ui message stream detection")
	}
	if err.ChunkType != "text-delta" || err.ChunkID != "part-1" {
		t.Fatalf("unexpected fields: %#v", err)
	}
}

func TestMissingToolResultsNoSuchToolAndNoOutputDetection(t *testing.T) {
	missing := NewMissingToolResultsError([]string{"call-1", "call-2"})
	noSuchTool := NewNoSuchToolError("weather", []string{"time", "search"})
	noOutput := NewNoOutputGeneratedError("", errors.New("empty response"))

	if !errors.Is(missing, ErrMissingToolResults) || !IsMissingToolResultsError(missing) {
		t.Fatalf("expected missing tool results detection")
	}
	if !errors.Is(noSuchTool, ErrNoSuchTool) || !IsNoSuchToolError(noSuchTool) {
		t.Fatalf("expected no such tool detection")
	}
	if !errors.Is(noOutput, ErrNoOutputGenerated) || !IsNoOutputGeneratedError(noOutput) {
		t.Fatalf("expected no output generated detection")
	}
	if missing.ToolCallIDs[1] != "call-2" {
		t.Fatalf("unexpected tool call ids: %#v", missing.ToolCallIDs)
	}
	if noSuchTool.ToolName != "weather" || noSuchTool.AvailableTools[1] != "search" {
		t.Fatalf("unexpected no such tool fields: %#v", noSuchTool)
	}
}

func TestTypedDetectionRecognizesExistingSDKErrorKinds(t *testing.T) {
	cases := []struct {
		name string
		err  error
		is   func(error) bool
	}{
		{name: "missing tool results", err: &SDKError{Kind: ErrMissingToolResults, Message: "legacy"}, is: IsMissingToolResultsError},
		{name: "invalid data content", err: &SDKError{Kind: ErrInvalidDataContent, Message: "legacy"}, is: IsInvalidDataContentError},
		{name: "invalid message role", err: &SDKError{Kind: ErrInvalidMessageRole, Message: "legacy"}, is: IsInvalidMessageRoleError},
		{name: "message conversion", err: &SDKError{Kind: ErrMessageConversion, Message: "legacy"}, is: IsMessageConversionError},
		{name: "no such tool", err: &SDKError{Kind: ErrNoSuchTool, Message: "legacy"}, is: IsNoSuchToolError},
		{name: "no output", err: &SDKError{Kind: ErrNoOutputGenerated, Message: "legacy"}, is: IsNoOutputGeneratedError},
		{name: "no object", err: &SDKError{Kind: ErrNoObjectGenerated, Message: "legacy"}, is: IsNoObjectGeneratedError},
		{name: "no image", err: &SDKError{Kind: ErrNoImageGenerated, Message: "legacy"}, is: IsNoImageGeneratedError},
		{name: "no speech", err: &SDKError{Kind: ErrNoSpeechGenerated, Message: "legacy"}, is: IsNoSpeechGeneratedError},
		{name: "no transcript", err: &SDKError{Kind: ErrNoTranscriptGenerated, Message: "legacy"}, is: IsNoTranscriptGeneratedError},
		{name: "no video", err: &SDKError{Kind: ErrNoVideoGenerated, Message: "legacy"}, is: IsNoVideoGeneratedError},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.is(fmt.Errorf("wrapped: %w", tc.err)) {
				t.Fatalf("expected %s detection for SDKError kind", tc.name)
			}
		})
	}
}
