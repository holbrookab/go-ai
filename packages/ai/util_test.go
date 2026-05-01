package ai

import (
	"bytes"
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"
)

func TestCosineSimilarity(t *testing.T) {
	got := CosineSimilarity([]float64{1, 0}, []float64{1, 0})
	if got != 1 {
		t.Fatalf("expected 1, got %v", got)
	}
	if !math.IsNaN(CosineSimilarity([]float64{1}, []float64{1, 2})) {
		t.Fatalf("expected NaN for mismatched dimensions")
	}
}

func TestParsePartialJSON(t *testing.T) {
	value, err := ParsePartialJSON(`{"name":"Ada"`)
	if err != nil {
		t.Fatalf("ParsePartialJSON failed: %v", err)
	}
	if value.(map[string]any)["name"] != "Ada" {
		t.Fatalf("unexpected value: %#v", value)
	}
}

func TestPrepareHeadersPreservesExplicitValues(t *testing.T) {
	got := PrepareHeaders(
		map[string]string{"content-type": "application/json"},
		map[string]string{"content-type": "text/plain", "x-default": "yes"},
	)
	if got["content-type"] != "application/json" || got["x-default"] != "yes" {
		t.Fatalf("unexpected headers: %#v", got)
	}
}

func TestMergeObjectsDeepMergesWithoutMutation(t *testing.T) {
	base := map[string]any{
		"nested": map[string]any{"a": "base", "b": "base"},
		"array":  []any{1.0},
	}
	overrides := map[string]any{
		"nested":       map[string]any{"b": "override"},
		"array":        []any{2.0},
		"__proto__":    "skip",
		"nil-override": nil,
	}
	got := MergeObjects(base, overrides)
	want := map[string]any{
		"nested": map[string]any{"a": "base", "b": "override"},
		"array":  []any{2.0},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected merge:\n got: %#v\nwant: %#v", got, want)
	}
	if base["nested"].(map[string]any)["b"] != "base" {
		t.Fatalf("base map was mutated: %#v", base)
	}
}

func TestConsumeReaderReportsErrors(t *testing.T) {
	sentinel := errors.New("boom")
	reader := failingReader{err: sentinel}
	var seen error
	err := ConsumeReader(context.Background(), reader, func(err error) {
		seen = err
	})
	if !errors.Is(err, sentinel) || !errors.Is(seen, sentinel) {
		t.Fatalf("expected reader error, got err=%v seen=%v", err, seen)
	}
	if err := ConsumeReader(context.Background(), bytes.NewBufferString("ok"), nil); err != nil {
		t.Fatalf("ConsumeReader failed: %v", err)
	}
}

func TestSerialJobExecutorRunsJobsInOrder(t *testing.T) {
	executor := NewSerialJobExecutor()
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	secondDone := make(chan struct{})
	var secondRan bool

	go func() {
		if err := executor.Run(context.Background(), func(context.Context) error {
			close(firstStarted)
			<-releaseFirst
			return nil
		}); err != nil {
			t.Errorf("Run failed: %v", err)
		}
	}()
	<-firstStarted

	go func() {
		defer close(secondDone)
		if err := executor.Run(context.Background(), func(context.Context) error {
			secondRan = true
			return nil
		}); err != nil {
			t.Errorf("Run failed: %v", err)
		}
	}()
	select {
	case <-secondDone:
		t.Fatalf("queued job ran before active job completed")
	default:
	}
	close(releaseFirst)
	<-secondDone
	if !secondRan {
		t.Fatalf("queued job did not run")
	}
}

func TestRetryWithExponentialBackoffRetriesRetryableErrors(t *testing.T) {
	attempts := 0
	var delays []time.Duration
	got, err := RetryWithExponentialBackoff(context.Background(), RetryOptions{
		MaxRetries:    2,
		InitialDelay:  100 * time.Millisecond,
		BackoffFactor: 2,
		Delay: func(_ context.Context, delay time.Duration) error {
			delays = append(delays, delay)
			return nil
		},
	}, func() (string, error) {
		attempts++
		if attempts < 3 {
			return "", NewAPICallError("rate limited", 429, map[string]string{"retry-after-ms": "5"}, true, nil)
		}
		return "ok", nil
	})
	if err != nil {
		t.Fatalf("RetryWithExponentialBackoff failed: %v", err)
	}
	if got != "ok" || attempts != 3 {
		t.Fatalf("unexpected retry result got=%q attempts=%d", got, attempts)
	}
	if !reflect.DeepEqual(delays, []time.Duration{5 * time.Millisecond, 5 * time.Millisecond}) {
		t.Fatalf("unexpected delays: %#v", delays)
	}
}

func TestRetryWithExponentialBackoffWrapsMaxRetries(t *testing.T) {
	_, err := RetryWithExponentialBackoff(context.Background(), RetryOptions{
		MaxRetries:   1,
		InitialDelay: time.Millisecond,
		Delay:        func(context.Context, time.Duration) error { return nil },
	}, func() (string, error) {
		return "", NewAPICallError("still rate limited", 429, nil, true, nil)
	})
	if !IsRetryError(err) {
		t.Fatalf("expected RetryError, got %T %v", err, err)
	}
	var retryErr *RetryError
	if !errors.As(err, &retryErr) || retryErr.Reason != RetryReasonMaxRetriesExceeded || len(retryErr.Errors) != 2 {
		t.Fatalf("unexpected retry error: %#v", retryErr)
	}
}

func TestStitchableStreamStitchesStreamsAndCloses(t *testing.T) {
	stitch := NewStitchableStream[int]()
	first := make(chan int, 2)
	first <- 1
	first <- 2
	close(first)
	second := make(chan int, 1)
	second <- 3
	close(second)

	if err := stitch.AddStream(first); err != nil {
		t.Fatalf("AddStream failed: %v", err)
	}
	if err := stitch.AddStream(second); err != nil {
		t.Fatalf("AddStream failed: %v", err)
	}
	stitch.Close()
	var got []int
	for value := range stitch.Stream() {
		got = append(got, value)
	}
	if !reflect.DeepEqual(got, []int{1, 2, 3}) {
		t.Fatalf("unexpected stitched stream: %#v", got)
	}
	if err := stitch.AddStream(first); err == nil {
		t.Fatalf("expected add after close to fail")
	}
}

type failingReader struct {
	err error
}

func (r failingReader) Read([]byte) (int, error) {
	return 0, r.err
}
