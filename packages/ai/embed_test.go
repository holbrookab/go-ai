package ai

import (
	"context"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestEmbedManyChunksAndAggregatesUsage(t *testing.T) {
	model := &mockEmbeddingModel{max: 2}
	result, err := EmbedMany(context.Background(), EmbedManyOptions{
		Model:  model,
		Values: []string{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("EmbedMany failed: %v", err)
	}
	if len(model.calls) != 2 {
		t.Fatalf("expected 2 model calls, got %d", len(model.calls))
	}
	seen := map[string]bool{}
	for _, call := range model.calls {
		seen[strings.Join(call, ",")] = true
	}
	if !seen["a,b"] || !seen["c"] {
		t.Fatalf("unexpected call chunks: %#v", model.calls)
	}
	if len(result.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result.Embeddings))
	}
	if !reflect.DeepEqual(result.Embeddings, [][]float64{{1}, {1}, {1}}) {
		t.Fatalf("embeddings should preserve input order, got %#v", result.Embeddings)
	}
	if result.Usage.Tokens == nil || *result.Usage.Tokens != 3 {
		t.Fatalf("expected 3 tokens, got %#v", result.Usage.Tokens)
	}
}

func TestEmbedReturnsSingleEmbedding(t *testing.T) {
	model := &mockEmbeddingModel{max: 10}
	result, err := Embed(context.Background(), EmbedOptions{
		Model: model,
		Value: "hello",
	})
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if !reflect.DeepEqual(result.Embedding, []float64{5}) {
		t.Fatalf("unexpected embedding: %#v", result.Embedding)
	}
}

func TestEmbedManyHonorsModelParallelSupport(t *testing.T) {
	model := &serialEmbeddingModel{mockEmbeddingModel: mockEmbeddingModel{max: 1}, sleep: 10 * time.Millisecond}
	if _, err := EmbedMany(context.Background(), EmbedManyOptions{
		Model:  model,
		Values: []string{"a", "b", "c"},
	}); err != nil {
		t.Fatalf("EmbedMany failed: %v", err)
	}
	if model.maxActive > 1 {
		t.Fatalf("expected serial embedding calls, saw %d concurrent calls", model.maxActive)
	}
}

type mockEmbeddingModel struct {
	max   int
	calls [][]string
}

func (m *mockEmbeddingModel) Provider() string { return "mock" }
func (m *mockEmbeddingModel) ModelID() string  { return "mock-embedding" }
func (m *mockEmbeddingModel) MaxEmbeddingsPerCall() int {
	return m.max
}

type serialEmbeddingModel struct {
	mockEmbeddingModel
	sleep     time.Duration
	mu        sync.Mutex
	active    int
	maxActive int
}

func (m *serialEmbeddingModel) SupportsParallelCalls() bool { return false }

func (m *serialEmbeddingModel) DoEmbed(_ context.Context, opts EmbeddingModelCallOptions) (*EmbeddingModelResult, error) {
	m.mu.Lock()
	m.active++
	if m.active > m.maxActive {
		m.maxActive = m.active
	}
	m.mu.Unlock()
	time.Sleep(m.sleep)
	m.mu.Lock()
	m.active--
	m.mu.Unlock()
	embeddings := make([][]float64, len(opts.Values))
	for i := range opts.Values {
		embeddings[i] = []float64{1}
	}
	return &EmbeddingModelResult{Embeddings: embeddings}, nil
}
func (m *mockEmbeddingModel) DoEmbed(_ context.Context, opts EmbeddingModelCallOptions) (*EmbeddingModelResult, error) {
	m.calls = append(m.calls, append([]string{}, opts.Values...))
	embeddings := make([][]float64, len(opts.Values))
	for i, value := range opts.Values {
		embeddings[i] = []float64{float64(len(value))}
	}
	tokens := len(opts.Values)
	return &EmbeddingModelResult{
		Embeddings: embeddings,
		Usage:      EmbeddingUsage{Tokens: &tokens},
	}, nil
}
