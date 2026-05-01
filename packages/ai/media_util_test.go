package ai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"reflect"
	"testing"
)

func TestSplitDataURLParsesBase64(t *testing.T) {
	parsed, err := SplitDataURL("data:image/png;base64,iVBORw0KGgo=")
	if err != nil {
		t.Fatalf("SplitDataURL failed: %v", err)
	}
	if parsed.MediaType != "image/png" {
		t.Fatalf("expected image/png media type, got %q", parsed.MediaType)
	}
	if !parsed.Base64 {
		t.Fatalf("expected base64 flag")
	}
	if !bytes.Equal(parsed.Data, []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}) {
		t.Fatalf("unexpected decoded bytes: %#v", parsed.Data)
	}
}

func TestConvertDataContentToBase64String(t *testing.T) {
	base64Text, err := ConvertDataContentToBase64String("aGVsbG8=")
	if err != nil {
		t.Fatalf("ConvertDataContentToBase64String string failed: %v", err)
	}
	if base64Text != "aGVsbG8=" {
		t.Fatalf("expected string content to pass through, got %q", base64Text)
	}

	encoded, err := ConvertDataContentToBase64String([]byte("hello"))
	if err != nil {
		t.Fatalf("ConvertDataContentToBase64String bytes failed: %v", err)
	}
	if encoded != "aGVsbG8=" {
		t.Fatalf("unexpected encoded bytes: %q", encoded)
	}
}

func TestConvertDataContentToBytesRejectsInvalidBase64(t *testing.T) {
	_, err := ConvertDataContentToBytes("not-base64")
	if !errors.Is(err, ErrInvalidDataContent) || !IsInvalidDataContentError(err) {
		t.Fatalf("expected invalid data content error, got %v", err)
	}
}

func TestConvertDataContentToBytesClonesByteSlices(t *testing.T) {
	input := []byte("hello")
	got, err := ConvertDataContentToBytes(input)
	if err != nil {
		t.Fatalf("ConvertDataContentToBytes failed: %v", err)
	}
	input[0] = 'j'
	if string(got) != "hello" {
		t.Fatalf("expected cloned bytes, got %q", string(got))
	}
}

func TestSplitDataURLParsesPercentEncodedText(t *testing.T) {
	parsed, err := SplitDataURL("data:text/plain,hello%20world")
	if err != nil {
		t.Fatalf("SplitDataURL failed: %v", err)
	}
	if parsed.MediaType != "text/plain" {
		t.Fatalf("expected text/plain media type, got %q", parsed.MediaType)
	}
	if parsed.Base64 {
		t.Fatalf("did not expect base64 flag")
	}
	if string(parsed.Data) != "hello world" {
		t.Fatalf("unexpected decoded text: %q", string(parsed.Data))
	}
}

func TestNormalizeFileDataFromBytesDetectsMediaType(t *testing.T) {
	data, mediaType, err := NormalizeFileData(context.Background(), FileDataInput{
		Data: []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'},
	}, nil)
	if err != nil {
		t.Fatalf("NormalizeFileData failed: %v", err)
	}
	if data.Type != FileDataTypeBytes {
		t.Fatalf("expected bytes data type, got %q", data.Type)
	}
	if mediaType != "image/png" {
		t.Fatalf("expected image/png media type, got %q", mediaType)
	}
}

func TestNormalizeFileDataFromBase64(t *testing.T) {
	data, mediaType, err := NormalizeFileData(context.Background(), FileDataInput{
		Base64:   "aGVsbG8=",
		Filename: "note.txt",
	}, nil)
	if err != nil {
		t.Fatalf("NormalizeFileData failed: %v", err)
	}
	if data.Type != FileDataTypeBytes || string(data.Data) != "hello" {
		t.Fatalf("unexpected data: %#v", data)
	}
	if mediaType != "text/plain" {
		t.Fatalf("expected filename media type fallback, got %q", mediaType)
	}
}

func TestNormalizeFileDataFromProviderReference(t *testing.T) {
	ref := ProviderReference{"openai": "file-123", "anthropic": "file-abc"}
	data, mediaType, err := NormalizeFileData(context.Background(), FileDataInput{
		ProviderReference: ref,
		MediaType:         "application/pdf",
	}, nil)
	if err != nil {
		t.Fatalf("NormalizeFileData failed: %v", err)
	}
	if data.Type != FileDataTypeReference {
		t.Fatalf("expected reference data type, got %q", data.Type)
	}
	if !reflect.DeepEqual(data.ProviderReference, ref) {
		t.Fatalf("unexpected provider reference: %#v", data.ProviderReference)
	}
	ref["openai"] = "changed"
	if data.ProviderReference["openai"] != "file-123" {
		t.Fatalf("expected provider reference to be cloned, got %#v", data.ProviderReference)
	}
	if mediaType != "application/pdf" {
		t.Fatalf("expected application/pdf, got %q", mediaType)
	}
}

func TestNormalizeFileDataRejectsInvalidBase64AsDataContent(t *testing.T) {
	_, _, err := NormalizeFileData(context.Background(), FileDataInput{Base64: "not-base64"}, nil)
	if !errors.Is(err, ErrInvalidDataContent) || !IsInvalidDataContentError(err) {
		t.Fatalf("expected invalid data content, got %v", err)
	}
}

func TestNormalizeFileDataFromText(t *testing.T) {
	data, mediaType, err := NormalizeFileData(context.Background(), FileDataInput{Text: "hello"}, nil)
	if err != nil {
		t.Fatalf("NormalizeFileData failed: %v", err)
	}
	if data.Type != FileDataTypeText || data.Text != "hello" {
		t.Fatalf("unexpected data: %#v", data)
	}
	if mediaType != "text/plain" {
		t.Fatalf("expected text/plain, got %q", mediaType)
	}
}

func TestNormalizeFileDataFromDataURL(t *testing.T) {
	data, mediaType, err := NormalizeFileData(context.Background(), FileDataInput{
		URL: "data:text/plain;base64,aGVsbG8=",
	}, nil)
	if err != nil {
		t.Fatalf("NormalizeFileData failed: %v", err)
	}
	if data.Type != FileDataTypeBytes || string(data.Data) != "hello" {
		t.Fatalf("unexpected data URL normalization: %#v", data)
	}
	if mediaType != "text/plain" {
		t.Fatalf("expected text/plain, got %q", mediaType)
	}
}

func TestNormalizeFileDataKeepsSupportedURL(t *testing.T) {
	data, mediaType, err := NormalizeFileData(context.Background(), FileDataInput{
		URL:       "https://cdn.example.com/image.png",
		MediaType: "image/png",
	}, &NormalizeFileDataOptions{
		SupportedURLs: map[string][]string{
			"image/*": {"https://cdn.example.com/*"},
		},
	})
	if err != nil {
		t.Fatalf("NormalizeFileData failed: %v", err)
	}
	if data.Type != FileDataTypeURL || data.URL != "https://cdn.example.com/image.png" {
		t.Fatalf("expected supported URL to remain by reference, got %#v", data)
	}
	if mediaType != "image/png" {
		t.Fatalf("expected image/png, got %q", mediaType)
	}
}

func TestNormalizeFileDataDownloadsUnsupportedURL(t *testing.T) {
	called := false
	data, mediaType, err := NormalizeFileData(context.Background(), FileDataInput{
		URL: "https://assets.example.com/photo",
	}, &NormalizeFileDataOptions{
		SupportedURLs: map[string][]string{
			"image/png": {"https://cdn.example.com/*"},
		},
		Download: func(_ context.Context, rawURL string) ([]byte, string, error) {
			called = true
			if rawURL != "https://assets.example.com/photo" {
				t.Fatalf("unexpected download URL: %q", rawURL)
			}
			return []byte{0xff, 0xd8, 0xff, 0xdb}, "image/jpeg", nil
		},
	})
	if err != nil {
		t.Fatalf("NormalizeFileData failed: %v", err)
	}
	if !called {
		t.Fatalf("expected download function to be called")
	}
	if data.Type != FileDataTypeBytes || !bytes.Equal(data.Data, []byte{0xff, 0xd8, 0xff, 0xdb}) {
		t.Fatalf("expected downloaded bytes, got %#v", data)
	}
	if mediaType != "image/jpeg" {
		t.Fatalf("expected image/jpeg, got %q", mediaType)
	}
}

func TestNormalizeFileDataRejectsAmbiguousSources(t *testing.T) {
	_, _, err := NormalizeFileData(context.Background(), FileDataInput{Text: "hello", URL: "https://example.com"}, nil)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("expected invalid argument, got %v", err)
	}
}

func TestIsURLSupportedMatchesMediaAndURLPatterns(t *testing.T) {
	supported := map[string][]string{
		"image/*":         {"https://images.example.com/*"},
		"application/pdf": {"/^https://docs[.]example[.]com/.+[.]pdf$/"},
	}
	if !IsURLSupported(supported, "image/png", "https://images.example.com/a/b.png") {
		t.Fatalf("expected wildcard URL support")
	}
	if !IsURLSupported(supported, "application/pdf", "https://docs.example.com/file.pdf") {
		t.Fatalf("expected regexp URL support")
	}
	if IsURLSupported(supported, "video/mp4", "https://images.example.com/a.mp4") {
		t.Fatalf("did not expect unsupported media type")
	}
	if IsURLSupported(supported, "image/png", "https://elsewhere.example.com/a.png") {
		t.Fatalf("did not expect unsupported URL")
	}
}

func TestNormalizeFilePartCarriesFilename(t *testing.T) {
	part, err := NormalizeFilePart(context.Background(), FileDataInput{
		Text:     "hello",
		Filename: "note.txt",
	}, nil)
	if err != nil {
		t.Fatalf("NormalizeFilePart failed: %v", err)
	}
	if part.Filename != "note.txt" {
		t.Fatalf("expected filename, got %q", part.Filename)
	}
	if part.MediaType != "text/plain" {
		t.Fatalf("expected text/plain, got %q", part.MediaType)
	}
}

func TestDownloadURLReturnsTypedStatusError(t *testing.T) {
	_, _, err := DownloadURL(context.Background(), "https://example.com/missing", &DownloadURLOptions{
		Client: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusNotFound,
				Status:     "404 Not Found",
				Header:     http.Header{},
				Body:       io.NopCloser(bytes.NewBuffer(nil)),
				Request:    req,
			}, nil
		})},
	})
	if !errors.Is(err, ErrDownload) || !IsDownloadError(err) {
		t.Fatalf("expected typed download error, got %v", err)
	}
	var downloadErr *DownloadError
	if !errors.As(err, &downloadErr) {
		t.Fatalf("expected DownloadError, got %T", err)
	}
	if downloadErr.URL != "https://example.com/missing" || downloadErr.StatusCode != http.StatusNotFound || downloadErr.StatusText != "Not Found" {
		t.Fatalf("unexpected download error: %#v", downloadErr)
	}
}

func TestDownloadURLRejectsUnsafeURL(t *testing.T) {
	_, _, err := DownloadURL(context.Background(), "http://127.0.0.1/file", nil)
	if !IsDownloadError(err) {
		t.Fatalf("expected download error for private URL, got %v", err)
	}
}

func TestDownloadURLEnforcesSizeLimit(t *testing.T) {
	_, _, err := DownloadURL(context.Background(), "https://example.com/large", &DownloadURLOptions{
		MaxBytes: 3,
		Client: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": []string{"text/plain"}},
				Body:       io.NopCloser(bytes.NewBufferString("hello")),
				Request:    req,
			}, nil
		})},
	})
	if !IsDownloadError(err) {
		t.Fatalf("expected size-limited download error, got %v", err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
