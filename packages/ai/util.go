package ai

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"time"
)

func CosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return math.NaN()
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return math.NaN()
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func DeepEqual(a, b any) bool {
	return reflect.DeepEqual(normalizeJSONComparable(a), normalizeJSONComparable(b))
}

func normalizeJSONComparable(v any) any {
	data, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		return v
	}
	return out
}

func HashString(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func FixPartialJSON(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return trimmed
	}
	openBraces := strings.Count(trimmed, "{") - strings.Count(trimmed, "}")
	openBrackets := strings.Count(trimmed, "[") - strings.Count(trimmed, "]")
	var b strings.Builder
	b.WriteString(trimmed)
	for i := 0; i < openBrackets; i++ {
		b.WriteByte(']')
	}
	for i := 0; i < openBraces; i++ {
		b.WriteByte('}')
	}
	return b.String()
}

func ParsePartialJSON(text string) (any, error) {
	var out any
	err := json.Unmarshal([]byte(FixPartialJSON(text)), &out)
	return out, err
}

func PrepareHeaders(headers map[string]string, defaultHeaders map[string]string) map[string]string {
	out := map[string]string{}
	for key, value := range headers {
		out[key] = value
	}
	for key, value := range defaultHeaders {
		if _, ok := out[key]; !ok {
			out[key] = value
		}
	}
	return out
}

func MergeObjects(base map[string]any, overrides map[string]any) map[string]any {
	if base == nil && overrides == nil {
		return nil
	}
	if base == nil {
		return cloneAnyMap(overrides)
	}
	if overrides == nil {
		return cloneAnyMap(base)
	}
	result := cloneAnyMap(base)
	for key, value := range overrides {
		if key == "__proto__" || key == "constructor" || key == "prototype" {
			continue
		}
		if value == nil {
			continue
		}
		baseMap, baseOK := result[key].(map[string]any)
		overrideMap, overrideOK := value.(map[string]any)
		if baseOK && overrideOK {
			result[key] = MergeObjects(baseMap, overrideMap)
			continue
		}
		result[key] = cloneJSONLike(value)
	}
	return result
}

func cloneAnyMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for key, value := range in {
		out[key] = cloneJSONLike(value)
	}
	return out
}

func cloneJSONLike(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneAnyMap(typed)
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = cloneJSONLike(item)
		}
		return out
	default:
		return value
	}
}

func ConsumeReader(ctx context.Context, reader io.Reader, onError func(error)) error {
	if ctx == nil {
		ctx = context.Background()
	}
	done := make(chan error, 1)
	go func() {
		_, err := io.Copy(io.Discard, reader)
		done <- err
	}()
	select {
	case <-ctx.Done():
		if onError != nil {
			onError(ctx.Err())
		}
		return ctx.Err()
	case err := <-done:
		if err != nil && onError != nil {
			onError(err)
		}
		return err
	}
}

type SerialJobExecutor struct {
	mu   sync.Mutex
	tail chan struct{}
}

func NewSerialJobExecutor() *SerialJobExecutor {
	done := make(chan struct{})
	close(done)
	return &SerialJobExecutor{tail: done}
}

func (e *SerialJobExecutor) Run(ctx context.Context, job func(context.Context) error) error {
	if e == nil {
		e = NewSerialJobExecutor()
	}
	if ctx == nil {
		ctx = context.Background()
	}
	e.mu.Lock()
	prev := e.tail
	next := make(chan struct{})
	e.tail = next
	e.mu.Unlock()

	select {
	case <-ctx.Done():
		close(next)
		return ctx.Err()
	case <-prev:
	}
	defer close(next)
	if job == nil {
		return nil
	}
	return job(ctx)
}

type RetryOptions struct {
	MaxRetries       int
	InitialDelay     time.Duration
	BackoffFactor    float64
	Delay            func(context.Context, time.Duration) error
	IsRetryable      func(error) bool
	Headers          func(error) map[string]string
	DisableWrapping  bool
	DisableRetryWhen func(error) bool
}

type retryableError interface {
	error
	IsRetryable() bool
	RetryHeaders() map[string]string
}

func RetryWithExponentialBackoff[T any](ctx context.Context, opts RetryOptions, fn func() (T, error)) (T, error) {
	var zero T
	if ctx == nil {
		ctx = context.Background()
	}
	maxRetries := opts.MaxRetries
	if maxRetries < 0 {
		return zero, &SDKError{Kind: ErrInvalidArgument, Message: "maxRetries must be >= 0"}
	}
	delay := opts.InitialDelay
	if delay <= 0 {
		delay = 2 * time.Second
	}
	backoffFactor := opts.BackoffFactor
	if backoffFactor <= 0 {
		backoffFactor = 2
	}
	delayFn := opts.Delay
	if delayFn == nil {
		delayFn = sleepDelay
	}
	var errorsSeen []error
	for {
		if err := ctx.Err(); err != nil {
			errorsSeen = append(errorsSeen, err)
			return zero, NewRetryError("Retry aborted.", RetryReasonAbort, errorsSeen)
		}
		value, err := fn()
		if err == nil {
			return value, nil
		}
		if ctx.Err() != nil {
			errorsSeen = append(errorsSeen, ctx.Err())
			return zero, NewRetryError("Retry aborted.", RetryReasonAbort, errorsSeen)
		}
		errorsSeen = append(errorsSeen, err)
		tryNumber := len(errorsSeen)
		if maxRetries == 0 {
			return zero, err
		}
		if opts.DisableRetryWhen != nil && opts.DisableRetryWhen(err) {
			if tryNumber == 1 || opts.DisableWrapping {
				return zero, err
			}
			return zero, NewRetryError(fmt.Sprintf("Failed after %d attempts with non-retryable error: %q", tryNumber, err.Error()), RetryReasonErrorNotRetryable, errorsSeen)
		}
		if !isRetryableForBackoff(err, opts) {
			if tryNumber == 1 || opts.DisableWrapping {
				return zero, err
			}
			return zero, NewRetryError(fmt.Sprintf("Failed after %d attempts with non-retryable error: %q", tryNumber, err.Error()), RetryReasonErrorNotRetryable, errorsSeen)
		}
		if tryNumber > maxRetries {
			return zero, NewRetryError(fmt.Sprintf("Failed after %d attempts. Last error: %s", tryNumber, err.Error()), RetryReasonMaxRetriesExceeded, errorsSeen)
		}
		wait := retryDelayFromHeaders(headersForRetryError(err, opts), delay)
		if err := delayFn(ctx, wait); err != nil {
			errorsSeen = append(errorsSeen, err)
			return zero, NewRetryError("Retry aborted.", RetryReasonAbort, errorsSeen)
		}
		delay = time.Duration(float64(delay) * backoffFactor)
	}
}

func isRetryableForBackoff(err error, opts RetryOptions) bool {
	if opts.IsRetryable != nil {
		return opts.IsRetryable(err)
	}
	if retryable, ok := err.(retryableError); ok {
		return retryable.IsRetryable()
	}
	return false
}

func headersForRetryError(err error, opts RetryOptions) map[string]string {
	if opts.Headers != nil {
		return opts.Headers(err)
	}
	if retryable, ok := err.(retryableError); ok {
		return retryable.RetryHeaders()
	}
	return nil
}

func retryDelayFromHeaders(headers map[string]string, fallback time.Duration) time.Duration {
	if len(headers) == 0 {
		return fallback
	}
	if value := firstHeaderValue(headers, "retry-after-ms"); value != "" {
		if parsed, err := time.ParseDuration(value + "ms"); err == nil && isReasonableRetryDelay(parsed, fallback) {
			return parsed
		}
	}
	if value := firstHeaderValue(headers, "retry-after"); value != "" {
		if seconds, err := time.ParseDuration(value + "s"); err == nil && isReasonableRetryDelay(seconds, fallback) {
			return seconds
		}
		if when, err := http.ParseTime(value); err == nil {
			delay := time.Until(when)
			if isReasonableRetryDelay(delay, fallback) {
				return delay
			}
		}
	}
	return fallback
}

func firstHeaderValue(headers map[string]string, key string) string {
	for header, value := range headers {
		if strings.EqualFold(header, key) {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func isReasonableRetryDelay(delay time.Duration, fallback time.Duration) bool {
	return delay >= 0 && (delay < time.Minute || delay < fallback)
}

func sleepDelay(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

type StitchableStream[T any] struct {
	stream  chan T
	actions chan stitchableAction[T]
	done    chan struct{}
	once    sync.Once
}

type stitchableAction[T any] struct {
	stream <-chan T
	close  bool
	stop   bool
	errc   chan error
}

func NewStitchableStream[T any]() *StitchableStream[T] {
	s := &StitchableStream[T]{
		stream:  make(chan T),
		actions: make(chan stitchableAction[T]),
		done:    make(chan struct{}),
	}
	go s.run()
	return s
}

func (s *StitchableStream[T]) Stream() <-chan T {
	if s == nil {
		closed := make(chan T)
		close(closed)
		return closed
	}
	return s.stream
}

func (s *StitchableStream[T]) AddStream(stream <-chan T) error {
	if s == nil {
		return &SDKError{Kind: ErrInvalidArgument, Message: "stitchable stream is nil"}
	}
	errc := make(chan error, 1)
	select {
	case <-s.done:
		return &SDKError{Kind: ErrInvalidArgument, Message: "cannot add stream: stitchable stream is closed"}
	case s.actions <- stitchableAction[T]{stream: stream, errc: errc}:
		return <-errc
	}
}

func (s *StitchableStream[T]) Close() {
	if s == nil {
		return
	}
	s.once.Do(func() {
		errc := make(chan error, 1)
		select {
		case <-s.done:
		case s.actions <- stitchableAction[T]{close: true, errc: errc}:
			<-errc
		}
	})
}

func (s *StitchableStream[T]) Terminate() {
	if s == nil {
		return
	}
	s.once.Do(func() {
		errc := make(chan error, 1)
		select {
		case <-s.done:
		case s.actions <- stitchableAction[T]{stop: true, errc: errc}:
			<-errc
		}
	})
}

func (s *StitchableStream[T]) run() {
	defer close(s.stream)
	defer close(s.done)
	queue := []<-chan T{}
	closed := false
	for {
		if len(queue) == 0 {
			if closed {
				return
			}
			action := <-s.actions
			if action.errc != nil {
				action.errc <- nil
			}
			if action.stop {
				return
			}
			if action.close {
				closed = true
				continue
			}
			if action.stream != nil {
				queue = append(queue, action.stream)
			}
			continue
		}
		select {
		case action := <-s.actions:
			accepted := true
			if closed && action.stream != nil {
				accepted = false
			}
			if action.stop {
				if action.errc != nil {
					action.errc <- nil
				}
				return
			}
			if action.close {
				if action.errc != nil {
					action.errc <- nil
				}
				closed = true
				continue
			}
			if accepted && action.stream != nil {
				queue = append(queue, action.stream)
			}
			if action.errc != nil {
				if accepted {
					action.errc <- nil
				} else {
					action.errc <- &SDKError{Kind: ErrInvalidArgument, Message: "cannot add stream: stitchable stream is closed"}
				}
			}
		case value, ok := <-queue[0]:
			if !ok {
				queue = queue[1:]
				continue
			}
			select {
			case action := <-s.actions:
				if action.stop {
					if action.errc != nil {
						action.errc <- nil
					}
					return
				}
				if action.close {
					closed = true
				} else if !closed && action.stream != nil {
					queue = append(queue, action.stream)
				}
				if action.errc != nil {
					if closed && action.stream != nil {
						action.errc <- &SDKError{Kind: ErrInvalidArgument, Message: "cannot add stream: stitchable stream is closed"}
					} else {
						action.errc <- nil
					}
				}
				queue[0] = prependValue(value, queue[0])
			case s.stream <- value:
			}
		}
	}
}

func prependValue[T any](value T, stream <-chan T) <-chan T {
	out := make(chan T)
	go func() {
		defer close(out)
		out <- value
		for item := range stream {
			out <- item
		}
	}()
	return out
}
