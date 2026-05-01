package ai

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestUploadFileNormalizesAndDelegates(t *testing.T) {
	suppressUploadWarnings(t)
	api := &mockFilesAPI{
		uploadFile: func(_ context.Context, opts UploadFileCallOptions) (*UploadFileModelResult, error) {
			if opts.Data.Type != UploadFileDataTypeData {
				t.Fatalf("expected data upload, got %q", opts.Data.Type)
			}
			if !reflect.DeepEqual(opts.Data.Data, []byte{1, 2, 3}) {
				t.Fatalf("unexpected data: %#v", opts.Data.Data)
			}
			if opts.MediaType != "application/octet-stream" {
				t.Fatalf("expected application/octet-stream media type, got %q", opts.MediaType)
			}
			if opts.Filename != "test.bin" {
				t.Fatalf("expected filename test.bin, got %q", opts.Filename)
			}
			if opts.Headers["X-Test"] != "true" {
				t.Fatalf("expected custom header to be forwarded")
			}
			if !strings.Contains(opts.Headers["User-Agent"], "go-ai/"+Version) {
				t.Fatalf("expected go-ai user agent, got %q", opts.Headers["User-Agent"])
			}
			if !reflect.DeepEqual(opts.ProviderOptions, ProviderOptions{"mock": map[string]any{"purpose": "assistants"}}) {
				t.Fatalf("unexpected provider options: %#v", opts.ProviderOptions)
			}
			return &UploadFileModelResult{
				ProviderReference: ProviderReference{"mock": "file-123"},
				MediaType:         "application/octet-stream",
				Filename:          "test.bin",
				ProviderMetadata:  ProviderMetadata{"mock": map[string]any{"size": 3}},
				Warnings:          []Warning{{Type: "unsupported", Feature: "filename"}},
				Request:           RequestMetadata{Body: map[string]any{"ok": true}},
				Response:          ResponseMetadata{ID: "response-1", ModelID: "files"},
			}, nil
		},
	}

	result, err := UploadFile(context.Background(), UploadFileOptions{
		API:             api,
		Data:            DataUploadFileData([]byte{1, 2, 3}),
		Filename:        "test.bin",
		Headers:         map[string]string{"X-Test": "true"},
		ProviderOptions: ProviderOptions{"mock": map[string]any{"purpose": "assistants"}},
	})
	if err != nil {
		t.Fatalf("UploadFile failed: %v", err)
	}
	if !reflect.DeepEqual(result.ProviderReference, ProviderReference{"mock": "file-123"}) {
		t.Fatalf("unexpected provider reference: %#v", result.ProviderReference)
	}
	if result.MediaType != "application/octet-stream" {
		t.Fatalf("unexpected media type: %q", result.MediaType)
	}
	if result.Response.ID != "response-1" {
		t.Fatalf("expected response metadata to be preserved")
	}
	if result.Request.Body == nil {
		t.Fatalf("expected request metadata to be preserved")
	}
}

func TestUploadFileDefaultsTextMediaType(t *testing.T) {
	api := &mockFilesAPI{
		uploadFile: func(_ context.Context, opts UploadFileCallOptions) (*UploadFileModelResult, error) {
			if opts.Data.Type != UploadFileDataTypeText || opts.Data.Text != "hello world" {
				t.Fatalf("unexpected upload data: %#v", opts.Data)
			}
			if opts.MediaType != "text/plain" {
				t.Fatalf("expected text/plain media type, got %q", opts.MediaType)
			}
			return &UploadFileModelResult{ProviderReference: ProviderReference{"mock": "file-text"}, Warnings: []Warning{}}, nil
		},
	}

	if _, err := UploadFile(context.Background(), UploadFileOptions{API: api, Data: TextUploadFileData("hello world")}); err != nil {
		t.Fatalf("UploadFile failed: %v", err)
	}
}

func TestUploadFilePassesBase64DataThrough(t *testing.T) {
	api := &mockFilesAPI{
		uploadFile: func(_ context.Context, opts UploadFileCallOptions) (*UploadFileModelResult, error) {
			if opts.Data.Base64 != "dGVzdA==" {
				t.Fatalf("expected base64 data to be preserved, got %#v", opts.Data)
			}
			if opts.MediaType != "text/plain" {
				t.Fatalf("expected detected text/plain media type, got %q", opts.MediaType)
			}
			return &UploadFileModelResult{ProviderReference: ProviderReference{"mock": "file-base64"}, Warnings: []Warning{}}, nil
		},
	}

	if _, err := UploadFile(context.Background(), UploadFileOptions{API: api, Data: Base64UploadFileData("dGVzdA==")}); err != nil {
		t.Fatalf("UploadFile failed: %v", err)
	}
}

func TestUploadFileResolvesFromProviderAndRetries(t *testing.T) {
	fail := errors.New("temporary")
	api := &mockFilesAPI{
		uploadFile: func(_ context.Context, _ UploadFileCallOptions) (*UploadFileModelResult, error) {
			return &UploadFileModelResult{ProviderReference: ProviderReference{"mock": "file-retry"}, Warnings: []Warning{}}, nil
		},
		errs: []error{fail},
	}
	provider := mockUploadProvider{files: api}
	maxRetries := 1

	result, err := UploadFile(context.Background(), UploadFileOptions{
		API:        provider,
		Data:       DataUploadFileData([]byte("hello")),
		MaxRetries: &maxRetries,
	})
	if err != nil {
		t.Fatalf("UploadFile failed: %v", err)
	}
	if api.calls != 2 {
		t.Fatalf("expected 2 calls after retry, got %d", api.calls)
	}
	if result.ProviderReference["mock"] != "file-retry" {
		t.Fatalf("unexpected provider reference: %#v", result.ProviderReference)
	}
}

func TestUploadFileEmitsCallbacks(t *testing.T) {
	api := &mockFilesAPI{
		uploadFile: func(_ context.Context, _ UploadFileCallOptions) (*UploadFileModelResult, error) {
			return &UploadFileModelResult{
				ProviderReference: ProviderReference{"mock": "file-123"},
				MediaType:         "text/plain",
				Filename:          "test.txt",
				Warnings:          []Warning{},
			}, nil
		},
	}
	var calls []string
	if _, err := UploadFile(context.Background(), UploadFileOptions{
		API:      api,
		Data:     TextUploadFileData("hello"),
		Filename: "test.txt",
		OnStart: func(event StartEvent) {
			calls = append(calls, event.Operation+":"+event.Provider+":"+event.ModelID)
		},
		OnFinish: func(event FinishEvent) {
			calls = append(calls, event.Operation)
		},
	}); err != nil {
		t.Fatalf("UploadFile failed: %v", err)
	}
	if !reflect.DeepEqual(calls, []string{"upload_file:mock:files", "upload_file"}) {
		t.Fatalf("unexpected callback calls: %#v", calls)
	}
}

func TestUploadFileRejectsProviderWithoutFiles(t *testing.T) {
	_, err := UploadFile(context.Background(), UploadFileOptions{
		API:  mockUploadProvider{},
		Data: DataUploadFileData([]byte{1}),
	})
	if !errors.Is(err, ErrUnsupportedFunction) {
		t.Fatalf("expected unsupported functionality error, got %v", err)
	}
	if !strings.Contains(err.Error(), "file uploads") {
		t.Fatalf("expected file upload support error, got %v", err)
	}
}

func TestUploadSkillNormalizesFilesAndDelegates(t *testing.T) {
	suppressUploadWarnings(t)
	api := &mockSkillsAPI{
		uploadSkill: func(_ context.Context, opts UploadSkillCallOptions) (*UploadSkillModelResult, error) {
			if len(opts.Files) != 2 {
				t.Fatalf("expected two files, got %d", len(opts.Files))
			}
			if opts.Files[0].Path != "skill.md" || opts.Files[0].Data.Text != "hello" {
				t.Fatalf("unexpected first file: %#v", opts.Files[0])
			}
			if opts.Files[1].Path != "data.bin" || !reflect.DeepEqual(opts.Files[1].Data.Data, []byte{1, 2}) {
				t.Fatalf("unexpected second file: %#v", opts.Files[1])
			}
			if opts.DisplayTitle != "My Skill" {
				t.Fatalf("unexpected display title: %q", opts.DisplayTitle)
			}
			if opts.ProviderOptions["mock"] == nil {
				t.Fatalf("expected provider options")
			}
			if !strings.Contains(opts.Headers["User-Agent"], "go-ai/"+Version) {
				t.Fatalf("expected go-ai user agent, got %q", opts.Headers["User-Agent"])
			}
			return &UploadSkillModelResult{
				ProviderReference: ProviderReference{"mock": "skill-123"},
				DisplayTitle:      "My Skill",
				Name:              "my-skill",
				Description:       "does things",
				LatestVersion:     "v1",
				ProviderMetadata:  ProviderMetadata{"mock": map[string]any{"status": "ok"}},
				Warnings:          []Warning{{Type: "unsupported", Feature: "displayTitle"}},
				Request:           RequestMetadata{Body: "request"},
				Response:          ResponseMetadata{ID: "response-2"},
			}, nil
		},
	}

	result, err := UploadSkill(context.Background(), UploadSkillOptions{
		API: api,
		Files: []UploadSkillFile{
			{Path: "skill.md", Data: TextUploadFileData("hello")},
			{Path: "data.bin", Data: DataUploadFileData([]byte{1, 2})},
		},
		DisplayTitle:    "My Skill",
		ProviderOptions: ProviderOptions{"mock": map[string]any{"custom": true}},
	})
	if err != nil {
		t.Fatalf("UploadSkill failed: %v", err)
	}
	if result.ProviderReference["mock"] != "skill-123" {
		t.Fatalf("unexpected provider reference: %#v", result.ProviderReference)
	}
	if result.Name != "my-skill" || result.LatestVersion != "v1" {
		t.Fatalf("unexpected skill metadata: %#v", result)
	}
	if result.Request.Body == nil || result.Response.ID != "response-2" {
		t.Fatalf("expected request/response metadata to be preserved")
	}
}

func TestUploadSkillResolvesFromProvider(t *testing.T) {
	api := &mockSkillsAPI{
		uploadSkill: func(_ context.Context, _ UploadSkillCallOptions) (*UploadSkillModelResult, error) {
			return &UploadSkillModelResult{ProviderReference: ProviderReference{"mock": "skill-123"}, Warnings: []Warning{}}, nil
		},
	}
	provider := mockUploadProvider{skills: api}

	if _, err := UploadSkill(context.Background(), UploadSkillOptions{
		API:   provider,
		Files: []UploadSkillFile{{Path: "skill.md", Data: TextUploadFileData("hello")}},
	}); err != nil {
		t.Fatalf("UploadSkill failed: %v", err)
	}
	if api.calls != 1 {
		t.Fatalf("expected upload skill call, got %d", api.calls)
	}
}

func TestUploadSkillEmitsCallbacks(t *testing.T) {
	api := &mockSkillsAPI{
		uploadSkill: func(_ context.Context, _ UploadSkillCallOptions) (*UploadSkillModelResult, error) {
			return &UploadSkillModelResult{
				ProviderReference: ProviderReference{"mock": "skill-123"},
				DisplayTitle:      "Skill",
				Warnings:          []Warning{},
			}, nil
		},
	}
	var calls []string
	if _, err := UploadSkill(context.Background(), UploadSkillOptions{
		API:          api,
		Files:        []UploadSkillFile{{Path: "skill.md", Data: TextUploadFileData("hello")}},
		DisplayTitle: "Skill",
		OnStart: func(event StartEvent) {
			calls = append(calls, event.Operation+":"+event.Provider+":"+event.ModelID)
		},
		OnFinish: func(event FinishEvent) {
			calls = append(calls, event.Operation)
		},
	}); err != nil {
		t.Fatalf("UploadSkill failed: %v", err)
	}
	if !reflect.DeepEqual(calls, []string{"upload_skill:mock:skills", "upload_skill"}) {
		t.Fatalf("unexpected callback calls: %#v", calls)
	}
}

func TestUploadSkillRejectsProviderWithoutSkills(t *testing.T) {
	_, err := UploadSkill(context.Background(), UploadSkillOptions{
		API:   mockUploadProvider{},
		Files: []UploadSkillFile{{Path: "skill.md", Data: TextUploadFileData("hello")}},
	})
	if !errors.Is(err, ErrUnsupportedFunction) {
		t.Fatalf("expected unsupported functionality error, got %v", err)
	}
	if !strings.Contains(err.Error(), "skills") {
		t.Fatalf("expected skills support error, got %v", err)
	}
}

type mockFilesAPI struct {
	calls      int
	errs       []error
	uploadFile func(context.Context, UploadFileCallOptions) (*UploadFileModelResult, error)
}

func (m *mockFilesAPI) Provider() string { return "mock" }

func (m *mockFilesAPI) UploadFile(ctx context.Context, opts UploadFileCallOptions) (*UploadFileModelResult, error) {
	m.calls++
	if len(m.errs) > 0 {
		err := m.errs[0]
		m.errs = m.errs[1:]
		if err != nil {
			return nil, err
		}
	}
	return m.uploadFile(ctx, opts)
}

type mockSkillsAPI struct {
	calls       int
	uploadSkill func(context.Context, UploadSkillCallOptions) (*UploadSkillModelResult, error)
}

func (m *mockSkillsAPI) Provider() string { return "mock" }

func (m *mockSkillsAPI) UploadSkill(ctx context.Context, opts UploadSkillCallOptions) (*UploadSkillModelResult, error) {
	m.calls++
	return m.uploadSkill(ctx, opts)
}

type mockUploadProvider struct {
	files  FilesAPI
	skills SkillsAPI
}

func (p mockUploadProvider) Provider() string  { return "mock" }
func (p mockUploadProvider) ModelID() string   { return "mock-model" }
func (p mockUploadProvider) Files() FilesAPI   { return p.files }
func (p mockUploadProvider) Skills() SkillsAPI { return p.skills }

func (p mockUploadProvider) SupportedURLs(context.Context) (map[string][]string, error) {
	return nil, nil
}

func (p mockUploadProvider) DoGenerate(context.Context, LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
	return nil, nil
}

func (p mockUploadProvider) DoStream(context.Context, LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
	return nil, nil
}

func suppressUploadWarnings(t *testing.T) {
	t.Helper()
	SetWarningLogger(WarningLoggerFunc(func([]Warning, string, string) {}))
	t.Cleanup(func() {
		SetWarningLogger(textWarningLogger{writer: os.Stderr})
	})
}
