package httputil

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type Doer interface {
	Do(*http.Request) (*http.Response, error)
}

func Client(doer Doer) Doer {
	if doer != nil {
		return doer
	}
	return http.DefaultClient
}

func PostJSON(ctx context.Context, client Doer, url string, headers map[string]string, body any) ([]byte, map[string]string, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		if v != "" {
			req.Header.Set(k, v)
		}
	}
	resp, err := Client(client).Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}
	responseHeaders := map[string]string{}
	for k, values := range resp.Header {
		if len(values) > 0 {
			responseHeaders[k] = values[0]
		}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, responseHeaders, fmt.Errorf("api request failed: status %d: %s", resp.StatusCode, string(data))
	}
	return data, responseHeaders, nil
}

func CloneHeaders(headers map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range headers {
		out[k] = v
	}
	return out
}
