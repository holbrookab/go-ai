package ai

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidStreamPart           = errors.New("invalid stream part")
	ErrInvalidToolApproval         = errors.New("invalid tool approval")
	ErrToolCallNotFoundForApproval = errors.New("tool call not found for approval")
	ErrToolCallRepair              = errors.New("tool call repair error")
	ErrUnsupportedModelVersion     = errors.New("unsupported model version")
	ErrUIMessageStream             = errors.New("ui message stream error")
)

type InvalidDataContentError struct {
	SDKError
	Content any
}

func NewInvalidDataContentError(content any, cause error, message string) *InvalidDataContentError {
	if message == "" {
		message = fmt.Sprintf("Invalid data content. Expected a base64 string or []byte, but got %T.", content)
	}
	return &InvalidDataContentError{
		SDKError: SDKError{Kind: ErrInvalidDataContent, Message: message, Cause: cause},
		Content:  content,
	}
}

func IsInvalidDataContentError(err error) bool {
	var target *InvalidDataContentError
	return errors.As(err, &target) || errors.Is(err, ErrInvalidDataContent)
}

type InvalidMessageRoleError struct {
	SDKError
	Role Role
}

func NewInvalidMessageRoleError(role Role) *InvalidMessageRoleError {
	return &InvalidMessageRoleError{
		SDKError: SDKError{
			Kind:    ErrInvalidMessageRole,
			Message: fmt.Sprintf("Invalid message role: %q. Must be one of: \"system\", \"user\", \"assistant\", \"tool\".", role),
		},
		Role: role,
	}
}

func IsInvalidMessageRoleError(err error) bool {
	var target *InvalidMessageRoleError
	return errors.As(err, &target) || errors.Is(err, ErrInvalidMessageRole)
}

type MessageConversionError struct {
	SDKError
	OriginalMessage any
}

func NewMessageConversionError(originalMessage any, message string) *MessageConversionError {
	return &MessageConversionError{
		SDKError:        SDKError{Kind: ErrMessageConversion, Message: message},
		OriginalMessage: originalMessage,
	}
}

func IsMessageConversionError(err error) bool {
	var target *MessageConversionError
	return errors.As(err, &target) || errors.Is(err, ErrMessageConversion)
}

type APICallError struct {
	SDKError
	StatusCode      int
	ResponseHeaders map[string]string
	Retryable       bool
}

func NewAPICallError(message string, statusCode int, headers map[string]string, retryable bool, cause error) *APICallError {
	if message == "" {
		message = fmt.Sprintf("API call failed with status %d.", statusCode)
	}
	return &APICallError{
		SDKError:        SDKError{Kind: ErrAPICall, Message: message, Cause: cause},
		StatusCode:      statusCode,
		ResponseHeaders: cloneStringMap(headers),
		Retryable:       retryable,
	}
}

func (e *APICallError) IsRetryable() bool {
	return e != nil && e.Retryable
}

func (e *APICallError) RetryHeaders() map[string]string {
	if e == nil {
		return nil
	}
	return cloneStringMap(e.ResponseHeaders)
}

func IsAPICallError(err error) bool {
	var target *APICallError
	return errors.As(err, &target) || errors.Is(err, ErrAPICall)
}

type DownloadError struct {
	SDKError
	URL        string
	StatusCode int
	StatusText string
}

func NewDownloadError(rawURL string, statusCode int, statusText string, message string, cause error) *DownloadError {
	if message == "" {
		if cause != nil {
			message = fmt.Sprintf("Failed to download %s: %v", rawURL, cause)
		} else {
			message = fmt.Sprintf("Failed to download %s: %d %s", rawURL, statusCode, statusText)
		}
	}
	return &DownloadError{
		SDKError:   SDKError{Kind: ErrDownload, Message: message, Cause: cause},
		URL:        rawURL,
		StatusCode: statusCode,
		StatusText: statusText,
	}
}

func IsDownloadError(err error) bool {
	var target *DownloadError
	return errors.As(err, &target) || errors.Is(err, ErrDownload)
}

type GatewayAuthenticationError struct {
	SDKError
}

func NewGatewayAuthenticationError(message string, cause error) *GatewayAuthenticationError {
	if message == "" {
		message = "Unauthenticated request to AI Gateway."
	}
	return &GatewayAuthenticationError{
		SDKError: SDKError{Kind: ErrGatewayAuthentication, Message: message, Cause: cause},
	}
}

func IsGatewayAuthenticationError(err error) bool {
	var target *GatewayAuthenticationError
	return errors.As(err, &target) || errors.Is(err, ErrGatewayAuthentication)
}

type GatewayError struct {
	SDKError
}

func NewGatewayError(message string, cause error) *GatewayError {
	if message == "" {
		message = "Gateway error."
	}
	return &GatewayError{
		SDKError: SDKError{Kind: ErrGateway, Message: message, Cause: cause},
	}
}

func IsGatewayError(err error) bool {
	var target *GatewayError
	return errors.As(err, &target) || errors.Is(err, ErrGateway)
}

func WrapGatewayError(err error) error {
	if err == nil || !IsGatewayAuthenticationError(err) {
		return err
	}
	return NewGatewayError(
		"Unauthenticated. Configure AI_GATEWAY_API_KEY or use a provider module. Learn more: https://ai-sdk.dev/unauthenticated-ai-gateway",
		err,
	)
}

type RetryErrorReason string

const (
	RetryReasonMaxRetriesExceeded RetryErrorReason = "maxRetriesExceeded"
	RetryReasonErrorNotRetryable  RetryErrorReason = "errorNotRetryable"
	RetryReasonAbort              RetryErrorReason = "abort"
)

type RetryError struct {
	SDKError
	Reason    RetryErrorReason
	LastError error
	Errors    []error
}

func NewRetryError(message string, reason RetryErrorReason, retryErrors []error) *RetryError {
	errorsCopy := append([]error(nil), retryErrors...)
	var last error
	if len(errorsCopy) > 0 {
		last = errorsCopy[len(errorsCopy)-1]
	}
	if message == "" {
		message = "Retry failed."
	}
	return &RetryError{
		SDKError:  SDKError{Kind: ErrRetry, Message: message, Cause: last},
		Reason:    reason,
		LastError: last,
		Errors:    errorsCopy,
	}
}

func IsRetryError(err error) bool {
	var target *RetryError
	return errors.As(err, &target) || errors.Is(err, ErrRetry)
}

func cloneStringMap(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

type InvalidStreamPartError struct {
	SDKError
	Chunk StreamPart
}

func NewInvalidStreamPartError(chunk StreamPart, message string) *InvalidStreamPartError {
	return &InvalidStreamPartError{
		SDKError: SDKError{Kind: ErrInvalidStreamPart, Message: message},
		Chunk:    chunk,
	}
}

func IsInvalidStreamPartError(err error) bool {
	var target *InvalidStreamPartError
	return errors.As(err, &target)
}

type InvalidToolApprovalError struct {
	SDKError
	ApprovalID string
}

func NewInvalidToolApprovalError(approvalID string) *InvalidToolApprovalError {
	return &InvalidToolApprovalError{
		SDKError: SDKError{
			Kind: ErrInvalidToolApproval,
			Message: fmt.Sprintf(
				"Tool approval response references unknown approvalId: %q. No matching tool-approval-request found in message history.",
				approvalID,
			),
		},
		ApprovalID: approvalID,
	}
}

func IsInvalidToolApprovalError(err error) bool {
	var target *InvalidToolApprovalError
	return errors.As(err, &target)
}

type ToolCallNotFoundForApprovalError struct {
	SDKError
	ToolCallID string
	ApprovalID string
}

func NewToolCallNotFoundForApprovalError(toolCallID, approvalID string) *ToolCallNotFoundForApprovalError {
	return &ToolCallNotFoundForApprovalError{
		SDKError: SDKError{
			Kind:    ErrToolCallNotFoundForApproval,
			Message: fmt.Sprintf("Tool call %q not found for approval request %q.", toolCallID, approvalID),
		},
		ToolCallID: toolCallID,
		ApprovalID: approvalID,
	}
}

func IsToolCallNotFoundForApprovalError(err error) bool {
	var target *ToolCallNotFoundForApprovalError
	return errors.As(err, &target)
}

type ToolCallRepairError struct {
	SDKError
	OriginalError error
}

func NewToolCallRepairError(cause error, originalError error) *ToolCallRepairError {
	return NewToolCallRepairErrorWithMessage("", cause, originalError)
}

func NewToolCallRepairErrorWithMessage(message string, cause error, originalError error) *ToolCallRepairError {
	if message == "" {
		message = fmt.Sprintf("Error repairing tool call: %v", cause)
	}
	return &ToolCallRepairError{
		SDKError:      SDKError{Kind: ErrToolCallRepair, Message: message, Cause: cause},
		OriginalError: originalError,
	}
}

func IsToolCallRepairError(err error) bool {
	var target *ToolCallRepairError
	return errors.As(err, &target)
}

type UnsupportedModelVersionError struct {
	SDKError
	Version  string
	Provider string
	ModelID  string
}

func NewUnsupportedModelVersionError(version, provider, modelID string) *UnsupportedModelVersionError {
	return &UnsupportedModelVersionError{
		SDKError: SDKError{
			Kind: ErrUnsupportedModelVersion,
			Message: fmt.Sprintf(
				"Unsupported model version %s for provider %q and model %q. AI SDK 5 only supports models that implement specification version \"v2\".",
				version,
				provider,
				modelID,
			),
		},
		Version:  version,
		Provider: provider,
		ModelID:  modelID,
	}
}

func IsUnsupportedModelVersionError(err error) bool {
	var target *UnsupportedModelVersionError
	return errors.As(err, &target)
}

type NoObjectGeneratedErrorOptions struct {
	Message      string
	Cause        error
	Text         string
	Response     ResponseMetadata
	Usage        Usage
	FinishReason FinishReason
}

type NoObjectGeneratedError struct {
	SDKError
	Text         string
	Response     ResponseMetadata
	Usage        Usage
	FinishReason FinishReason
}

func NewNoObjectGeneratedError(opts NoObjectGeneratedErrorOptions) *NoObjectGeneratedError {
	if opts.Message == "" {
		opts.Message = "No object generated."
	}
	return &NoObjectGeneratedError{
		SDKError:     SDKError{Kind: ErrNoObjectGenerated, Message: opts.Message, Cause: opts.Cause},
		Text:         opts.Text,
		Response:     opts.Response,
		Usage:        opts.Usage,
		FinishReason: opts.FinishReason,
	}
}

func IsNoObjectGeneratedError(err error) bool {
	var target *NoObjectGeneratedError
	return errors.As(err, &target) || errors.Is(err, ErrNoObjectGenerated)
}

type NoImageGeneratedErrorOptions struct {
	Message   string
	Cause     error
	Responses []ResponseMetadata
}

type NoImageGeneratedError struct {
	SDKError
	Responses []ResponseMetadata
}

func NewNoImageGeneratedError(opts NoImageGeneratedErrorOptions) *NoImageGeneratedError {
	if opts.Message == "" {
		opts.Message = "No image generated."
	}
	return &NoImageGeneratedError{
		SDKError:  SDKError{Kind: ErrNoImageGenerated, Message: opts.Message, Cause: opts.Cause},
		Responses: opts.Responses,
	}
}

func IsNoImageGeneratedError(err error) bool {
	var target *NoImageGeneratedError
	return errors.As(err, &target) || errors.Is(err, ErrNoImageGenerated)
}

type NoSpeechGeneratedError struct {
	SDKError
	Responses []ResponseMetadata
}

func NewNoSpeechGeneratedError(responses []ResponseMetadata) *NoSpeechGeneratedError {
	return &NoSpeechGeneratedError{
		SDKError:  SDKError{Kind: ErrNoSpeechGenerated, Message: "No speech audio generated."},
		Responses: responses,
	}
}

func IsNoSpeechGeneratedError(err error) bool {
	var target *NoSpeechGeneratedError
	return errors.As(err, &target) || errors.Is(err, ErrNoSpeechGenerated)
}

type NoTranscriptGeneratedError struct {
	SDKError
	Responses []ResponseMetadata
}

func NewNoTranscriptGeneratedError(responses []ResponseMetadata) *NoTranscriptGeneratedError {
	return &NoTranscriptGeneratedError{
		SDKError:  SDKError{Kind: ErrNoTranscriptGenerated, Message: "No transcript generated."},
		Responses: responses,
	}
}

func IsNoTranscriptGeneratedError(err error) bool {
	var target *NoTranscriptGeneratedError
	return errors.As(err, &target) || errors.Is(err, ErrNoTranscriptGenerated)
}

type NoVideoGeneratedErrorOptions struct {
	Message   string
	Cause     error
	Responses []ResponseMetadata
}

type NoVideoGeneratedError struct {
	SDKError
	Responses []ResponseMetadata
}

func NewNoVideoGeneratedError(opts NoVideoGeneratedErrorOptions) *NoVideoGeneratedError {
	if opts.Message == "" {
		opts.Message = "No video generated."
	}
	return &NoVideoGeneratedError{
		SDKError:  SDKError{Kind: ErrNoVideoGenerated, Message: opts.Message, Cause: opts.Cause},
		Responses: opts.Responses,
	}
}

func IsNoVideoGeneratedError(err error) bool {
	var target *NoVideoGeneratedError
	return errors.As(err, &target) || errors.Is(err, ErrNoVideoGenerated)
}

type UIMessageStreamError struct {
	SDKError
	ChunkType string
	ChunkID   string
}

func NewUIMessageStreamError(chunkType, chunkID, message string) *UIMessageStreamError {
	return &UIMessageStreamError{
		SDKError:  SDKError{Kind: ErrUIMessageStream, Message: message},
		ChunkType: chunkType,
		ChunkID:   chunkID,
	}
}

func IsUIMessageStreamError(err error) bool {
	var target *UIMessageStreamError
	return errors.As(err, &target)
}

type MissingToolResultsError struct {
	SDKError
	ToolCallIDs []string
}

func NewMissingToolResultsError(toolCallIDs []string) *MissingToolResultsError {
	plural := len(toolCallIDs) > 1
	resultVerb := "is"
	callSuffix := ""
	if plural {
		resultVerb = "are"
		callSuffix = "s"
	}
	return &MissingToolResultsError{
		SDKError: SDKError{
			Kind:    ErrMissingToolResults,
			Message: fmt.Sprintf("Tool result%s %s missing for tool call%s %s.", callSuffix, resultVerb, callSuffix, joinComma(toolCallIDs)),
		},
		ToolCallIDs: append([]string(nil), toolCallIDs...),
	}
}

func IsMissingToolResultsError(err error) bool {
	var target *MissingToolResultsError
	return errors.As(err, &target) || errors.Is(err, ErrMissingToolResults)
}

type NoSuchToolError struct {
	SDKError
	ToolName       string
	AvailableTools []string
}

func NewNoSuchToolError(toolName string, availableTools []string) *NoSuchToolError {
	availability := "No tools are available."
	if availableTools != nil {
		availability = fmt.Sprintf("Available tools: %s.", joinComma(availableTools))
	}
	return &NoSuchToolError{
		SDKError: SDKError{
			Kind:    ErrNoSuchTool,
			Message: fmt.Sprintf("Model tried to call unavailable tool %q. %s", toolName, availability),
		},
		ToolName:       toolName,
		AvailableTools: append([]string(nil), availableTools...),
	}
}

func IsNoSuchToolError(err error) bool {
	var target *NoSuchToolError
	return errors.As(err, &target) || errors.Is(err, ErrNoSuchTool)
}

type NoOutputGeneratedError struct {
	SDKError
}

func NewNoOutputGeneratedError(message string, cause error) *NoOutputGeneratedError {
	if message == "" {
		message = "No output generated."
	}
	return &NoOutputGeneratedError{
		SDKError: SDKError{Kind: ErrNoOutputGenerated, Message: message, Cause: cause},
	}
}

func IsNoOutputGeneratedError(err error) bool {
	var target *NoOutputGeneratedError
	return errors.As(err, &target) || errors.Is(err, ErrNoOutputGenerated)
}

func joinComma(values []string) string {
	if len(values) == 0 {
		return ""
	}
	out := values[0]
	for _, value := range values[1:] {
		out += ", " + value
	}
	return out
}
