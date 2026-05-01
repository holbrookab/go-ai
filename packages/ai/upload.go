package ai

import (
	"context"
	"encoding/base64"

	"github.com/holbrookab/go-ai/internal/retry"
)

const (
	UploadFileDataTypeData = "data"
	UploadFileDataTypeText = "text"
)

type ProviderReference map[string]string

type FilesAPI interface {
	Provider() string
	UploadFile(ctx context.Context, opts UploadFileCallOptions) (*UploadFileModelResult, error)
}

type FilesProvider interface {
	Files() FilesAPI
}

type SkillsAPI interface {
	Provider() string
	UploadSkill(ctx context.Context, opts UploadSkillCallOptions) (*UploadSkillModelResult, error)
}

type SkillsProvider interface {
	Skills() SkillsAPI
}

type FilesAPIResolver interface {
	FilesAPI(ref string) (FilesAPI, error)
}

type SkillsAPIResolver interface {
	SkillsAPI(ref string) (SkillsAPI, error)
}

type UploadFileData struct {
	Type   string
	Data   []byte
	Base64 string
	Text   string
}

func DataUploadFileData(data []byte) UploadFileData {
	return UploadFileData{Type: UploadFileDataTypeData, Data: cloneBytes(data)}
}

func Base64UploadFileData(data string) UploadFileData {
	return UploadFileData{Type: UploadFileDataTypeData, Base64: data}
}

func TextUploadFileData(text string) UploadFileData {
	return UploadFileData{Type: UploadFileDataTypeText, Text: text}
}

type UploadFileCallOptions struct {
	Data            UploadFileData
	MediaType       string
	Filename        string
	ProviderOptions ProviderOptions
	Headers         map[string]string
}

type UploadFileModelResult struct {
	ProviderReference ProviderReference
	MediaType         string
	Filename          string
	ProviderMetadata  ProviderMetadata
	Warnings          []Warning
	Request           RequestMetadata
	Response          ResponseMetadata
}

type UploadFileOptions struct {
	API              any
	Data             UploadFileData
	MediaType        string
	Filename         string
	MaxRetries       *int
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type UploadFileResult struct {
	ProviderReference ProviderReference
	MediaType         string
	Filename          string
	ProviderMetadata  ProviderMetadata
	Warnings          []Warning
	Request           RequestMetadata
	Response          ResponseMetadata
}

type UploadSkillFile struct {
	Path string
	Data UploadFileData
}

type UploadSkillCallOptions struct {
	Files           []UploadSkillFile
	DisplayTitle    string
	ProviderOptions ProviderOptions
	Headers         map[string]string
}

type UploadSkillModelResult struct {
	ProviderReference ProviderReference
	DisplayTitle      string
	Name              string
	Description       string
	LatestVersion     string
	ProviderMetadata  ProviderMetadata
	Warnings          []Warning
	Request           RequestMetadata
	Response          ResponseMetadata
}

type UploadSkillOptions struct {
	API              any
	Files            []UploadSkillFile
	DisplayTitle     string
	MaxRetries       *int
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	Telemetry        Telemetry
	TelemetryOptions TelemetryOptions
	OnStart          func(StartEvent)
	OnFinish         func(FinishEvent)
	OnError          func(ErrorEvent)
}

type UploadSkillResult struct {
	ProviderReference ProviderReference
	DisplayTitle      string
	Name              string
	Description       string
	LatestVersion     string
	ProviderMetadata  ProviderMetadata
	Warnings          []Warning
	Request           RequestMetadata
	Response          ResponseMetadata
}

func UploadFile(ctx context.Context, opts UploadFileOptions) (uploadResult *UploadFileResult, err error) {
	api, err := ResolveFilesAPI(opts.API, nil)
	if err != nil {
		return nil, err
	}
	eventModel := apiEventModel{provider: api.Provider(), modelID: "files"}
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventUploadFileStart, OperationUploadFile, eventModel, map[string]any{
		"media_type": opts.MediaType,
		"filename":   opts.Filename,
		"data_type":  opts.Data.Type,
	})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventUploadFileError, OperationUploadFile, err)
		}
	}()
	data, mediaType, err := normalizeUploadFileData(opts.Data, opts.MediaType, opts.Filename)
	if err != nil {
		return nil, err
	}
	callOptions := UploadFileCallOptions{
		Data:            data,
		MediaType:       mediaType,
		Filename:        opts.Filename,
		ProviderOptions: opts.ProviderOptions,
		Headers:         withUserAgent(opts.Headers, "go-ai/"+Version),
	}

	maxRetries := 2
	if opts.MaxRetries != nil {
		maxRetries = *opts.MaxRetries
	}
	var modelResult *UploadFileModelResult
	if err := retry.Do(ctx, maxRetries, func() error {
		result, err := api.UploadFile(ctx, callOptions)
		if err != nil {
			return err
		}
		modelResult = result
		return nil
	}); err != nil {
		return nil, err
	}
	if modelResult == nil {
		return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "files api returned nil upload file result"}
	}
	if len(modelResult.ProviderReference) == 0 {
		return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "files api returned no provider reference"}
	}
	LogWarnings(modelResult.Warnings, api.Provider(), "")
	uploadResult = &UploadFileResult{
		ProviderReference: modelResult.ProviderReference,
		MediaType:         modelResult.MediaType,
		Filename:          modelResult.Filename,
		ProviderMetadata:  modelResult.ProviderMetadata,
		Warnings:          modelResult.Warnings,
		Request:           modelResult.Request,
		Response:          modelResult.Response,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventUploadFileFinish, OperationUploadFile, uploadResult, map[string]any{
		"media_type": uploadResult.MediaType,
		"filename":   uploadResult.Filename,
	})
	return uploadResult, nil
}

func UploadSkill(ctx context.Context, opts UploadSkillOptions) (uploadResult *UploadSkillResult, err error) {
	api, err := ResolveSkillsAPI(opts.API, nil)
	if err != nil {
		return nil, err
	}
	eventModel := apiEventModel{provider: api.Provider(), modelID: "skills"}
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventUploadSkillStart, OperationUploadSkill, eventModel, map[string]any{
		"display_title": opts.DisplayTitle,
		"file_count":    len(opts.Files),
	})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventUploadSkillError, OperationUploadSkill, err)
		}
	}()
	files := make([]UploadSkillFile, len(opts.Files))
	for i, file := range opts.Files {
		data, _, err := normalizeUploadFileData(file.Data, "", "")
		if err != nil {
			return nil, err
		}
		files[i] = UploadSkillFile{Path: file.Path, Data: data}
	}
	callOptions := UploadSkillCallOptions{
		Files:           files,
		DisplayTitle:    opts.DisplayTitle,
		ProviderOptions: opts.ProviderOptions,
		Headers:         withUserAgent(opts.Headers, "go-ai/"+Version),
	}

	maxRetries := 2
	if opts.MaxRetries != nil {
		maxRetries = *opts.MaxRetries
	}
	var modelResult *UploadSkillModelResult
	if err := retry.Do(ctx, maxRetries, func() error {
		result, err := api.UploadSkill(ctx, callOptions)
		if err != nil {
			return err
		}
		modelResult = result
		return nil
	}); err != nil {
		return nil, err
	}
	if modelResult == nil {
		return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "skills api returned nil upload skill result"}
	}
	if len(modelResult.ProviderReference) == 0 {
		return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "skills api returned no provider reference"}
	}
	LogWarnings(modelResult.Warnings, api.Provider(), "")
	uploadResult = &UploadSkillResult{
		ProviderReference: modelResult.ProviderReference,
		DisplayTitle:      modelResult.DisplayTitle,
		Name:              modelResult.Name,
		Description:       modelResult.Description,
		LatestVersion:     modelResult.LatestVersion,
		ProviderMetadata:  modelResult.ProviderMetadata,
		Warnings:          modelResult.Warnings,
		Request:           modelResult.Request,
		Response:          modelResult.Response,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventUploadSkillFinish, OperationUploadSkill, uploadResult, map[string]any{
		"display_title":  uploadResult.DisplayTitle,
		"name":           uploadResult.Name,
		"latest_version": uploadResult.LatestVersion,
	})
	return uploadResult, nil
}

type apiEventModel struct {
	provider string
	modelID  string
}

func (m apiEventModel) Provider() string { return m.provider }
func (m apiEventModel) ModelID() string  { return m.modelID }

func ResolveFilesAPI(api any, resolver FilesAPIResolver) (FilesAPI, error) {
	switch api := api.(type) {
	case FilesAPI:
		if isNil(api) {
			return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "files api is required"}
		}
		return api, nil
	case FilesProvider:
		files := api.Files()
		if isNil(files) {
			return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "The provider does not support file uploads. Make sure it exposes a Files method."}
		}
		return files, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "files api resolver is required"}
		}
		files, err := resolver.FilesAPI(api)
		if err != nil {
			return nil, err
		}
		if isNil(files) {
			return nil, noSuchModelError("files api", api)
		}
		return files, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "api is required"}
	default:
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "The provider does not support file uploads. Make sure it exposes a Files method."}
	}
}

func ResolveSkillsAPI(api any, resolver SkillsAPIResolver) (SkillsAPI, error) {
	switch api := api.(type) {
	case SkillsAPI:
		if isNil(api) {
			return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "skills api is required"}
		}
		return api, nil
	case SkillsProvider:
		skills := api.Skills()
		if isNil(skills) {
			return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "The provider does not support skills. Make sure it exposes a Skills method."}
		}
		return skills, nil
	case string:
		if resolver == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "skills api resolver is required"}
		}
		skills, err := resolver.SkillsAPI(api)
		if err != nil {
			return nil, err
		}
		if isNil(skills) {
			return nil, noSuchModelError("skills api", api)
		}
		return skills, nil
	case nil:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "api is required"}
	default:
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "The provider does not support skills. Make sure it exposes a Skills method."}
	}
}

func (r *ProviderRegistry) FilesAPI(ref string) (FilesAPI, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	if modelID != "files" {
		return nil, noSuchModelError("files api", ref)
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	filesProvider, ok := provider.(FilesProvider)
	if !ok {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "provider does not support file uploads"}
	}
	files := filesProvider.Files()
	if isNil(files) {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "provider does not support file uploads"}
	}
	return files, nil
}

func (r *ProviderRegistry) SkillsAPI(ref string) (SkillsAPI, error) {
	providerName, modelID, err := r.splitProviderModelRef(ref)
	if err != nil {
		return nil, err
	}
	if modelID != "skills" {
		return nil, noSuchModelError("skills api", ref)
	}
	r.mu.RLock()
	provider := r.providers[providerName]
	r.mu.RUnlock()
	if provider == nil {
		return nil, &SDKError{Kind: ErrNoSuchProvider, Message: providerName}
	}
	skillsProvider, ok := provider.(SkillsProvider)
	if !ok {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "provider does not support skills"}
	}
	skills := skillsProvider.Skills()
	if isNil(skills) {
		return nil, &SDKError{Kind: ErrUnsupportedFunction, Message: "provider does not support skills"}
	}
	return skills, nil
}

func normalizeUploadFileData(data UploadFileData, mediaType string, filename string) (UploadFileData, string, error) {
	switch data.Type {
	case UploadFileDataTypeText:
		if data.Data != nil || data.Base64 != "" {
			return UploadFileData{}, "", &SDKError{Kind: ErrInvalidArgument, Message: "text upload data must not include binary data"}
		}
		if mediaType == "" {
			mediaType = "text/plain"
		}
		return UploadFileData{Type: UploadFileDataTypeText, Text: data.Text}, normalizeMediaType(mediaType), nil
	case UploadFileDataTypeData:
		if data.Data != nil && data.Base64 != "" {
			return UploadFileData{}, "", &SDKError{Kind: ErrInvalidArgument, Message: "data upload data must include bytes or base64, not both"}
		}
		out := UploadFileData{Type: UploadFileDataTypeData, Data: cloneBytes(data.Data), Base64: data.Base64}
		if mediaType == "" {
			detectedMediaType, err := detectUploadFileMediaType(out, filename)
			if err != nil {
				return UploadFileData{}, "", err
			}
			mediaType = detectedMediaType
		}
		return out, normalizeMediaType(mediaType), nil
	default:
		return UploadFileData{}, "", &SDKError{Kind: ErrInvalidArgument, Message: "upload data type must be data or text"}
	}
}

func detectUploadFileMediaType(data UploadFileData, filename string) (string, error) {
	if data.Base64 != "" {
		decoded, err := base64.StdEncoding.DecodeString(data.Base64)
		if err != nil {
			return "", &SDKError{Kind: ErrInvalidArgument, Message: "upload data contains invalid base64 data", Cause: err}
		}
		if mediaType := DetectMediaType(decoded, filename); mediaType != "" && mediaType != "application/octet-stream" {
			return mediaType, nil
		}
		if isLikelyTextBytes(decoded) {
			return "text/plain", nil
		}
		if mediaType := DetectMediaType(nil, filename); mediaType != "" {
			return mediaType, nil
		}
		return "application/octet-stream", nil
	}
	if mediaType := DetectMediaType(data.Data, filename); mediaType != "" && mediaType != "application/octet-stream" {
		return mediaType, nil
	}
	if isLikelyTextBytes(data.Data) {
		return "text/plain", nil
	}
	return "application/octet-stream", nil
}

func isLikelyTextBytes(data []byte) bool {
	const checkLength = 512
	if len(data) == 0 {
		return false
	}
	if len(data) > checkLength {
		data = data[:checkLength]
	}
	for _, b := range data {
		if b == 0x00 || (b < 0x20 && b != 0x09 && b != 0x0a && b != 0x0d) {
			return false
		}
	}
	return true
}
