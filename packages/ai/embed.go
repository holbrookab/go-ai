package ai

import (
	"context"
	"sync"

	"github.com/holbrookab/go-ai/internal/retry"
)

func Embed(ctx context.Context, opts EmbedOptions) (embedResult *EmbedResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventEmbedStart, OperationEmbed, opts.Model, map[string]any{"value_count": 1, "input.value": opts.Value})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventEmbedError, OperationEmbed, err)
		}
	}()
	if opts.Model == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "model is required"}
	}
	result, err := EmbedMany(ctx, EmbedManyOptions{
		Model:            opts.Model,
		Values:           []string{opts.Value},
		MaxRetries:       opts.MaxRetries,
		Headers:          opts.Headers,
		ProviderOptions:  opts.ProviderOptions,
		TelemetryOptions: opts.TelemetryOptions,
	})
	if err != nil {
		return nil, err
	}
	if len(result.Embeddings) == 0 {
		return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "model returned no embedding"}
	}
	response := ResponseMetadata{}
	if len(result.Responses) > 0 {
		response = result.Responses[0]
	}
	embedResult = &EmbedResult{
		Value:            opts.Value,
		Embedding:        result.Embeddings[0],
		Usage:            result.Usage,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Response:         response,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventEmbedFinish, OperationEmbed, embedResult, map[string]any{
		"usage":            embedResult.Usage,
		"output.embedding": embedResult.Embedding,
	})
	return embedResult, nil
}

func EmbedMany(ctx context.Context, opts EmbedManyOptions) (embedManyResult *EmbedManyResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventEmbedManyStart, OperationEmbedMany, opts.Model, map[string]any{"value_count": len(opts.Values), "input.values": append([]string(nil), opts.Values...)})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventEmbedManyError, OperationEmbedMany, err)
		}
	}()
	if opts.Model == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "model is required"}
	}
	values := append([]string{}, opts.Values...)
	if len(values) == 0 {
		embedManyResult = &EmbedManyResult{Values: values}
		emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventEmbedManyFinish, OperationEmbedMany, embedManyResult, map[string]any{"value_count": 0})
		return embedManyResult, nil
	}

	chunks := embeddingChunks(values, opts.Model.MaxEmbeddingsPerCall())
	maxParallel := opts.MaxParallelCalls
	if maxParallel <= 0 || maxParallel > len(chunks) {
		maxParallel = len(chunks)
	}
	if supports, ok := opts.Model.(interface{ SupportsParallelCalls() bool }); ok && !supports.SupportsParallelCalls() {
		maxParallel = 1
	}
	maxRetries := 2
	if opts.MaxRetries != nil {
		maxRetries = *opts.MaxRetries
	}

	type chunkResult struct {
		index  int
		result *EmbeddingModelResult
		err    error
	}
	sem := make(chan struct{}, maxParallel)
	results := make(chan chunkResult, len(chunks))
	var wg sync.WaitGroup
	for i, chunk := range chunks {
		sem <- struct{}{}
		wg.Add(1)
		go func(index int, values []string) {
			defer wg.Done()
			defer func() { <-sem }()
			var modelResult *EmbeddingModelResult
			err := retry.Do(ctx, maxRetries, func() error {
				result, err := opts.Model.DoEmbed(ctx, EmbeddingModelCallOptions{
					Values:          values,
					ProviderOptions: opts.ProviderOptions,
					Headers:         withUserAgent(opts.Headers, "go-ai/"+Version),
				})
				if err != nil {
					return err
				}
				modelResult = result
				return nil
			})
			results <- chunkResult{index: index, result: modelResult, err: err}
		}(i, chunk)
	}
	wg.Wait()
	close(results)

	chunkResults := make([]*EmbeddingModelResult, len(chunks))
	for result := range results {
		if result.err != nil {
			return nil, result.err
		}
		if result.result == nil {
			return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "model returned nil result"}
		}
		chunkResults[result.index] = result.result
	}

	embeddings := make([][]float64, 0, len(values))
	usage := EmbeddingUsage{}
	var warnings []Warning
	var responses []ResponseMetadata
	var metadata ProviderMetadata
	for _, result := range chunkResults {
		if len(result.Embeddings) == 0 {
			return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "model returned no embeddings"}
		}
		embeddings = append(embeddings, result.Embeddings...)
		usage = AddEmbeddingUsage(usage, result.Usage)
		warnings = append(warnings, result.Warnings...)
		responses = append(responses, result.Response)
		metadata = mergeMetadata(metadata, result.ProviderMetadata)
	}
	LogWarnings(warnings, opts.Model.Provider(), opts.Model.ModelID())
	if len(embeddings) != len(values) {
		return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "model returned a different number of embeddings"}
	}

	embedManyResult = &EmbedManyResult{
		Values:           values,
		Embeddings:       embeddings,
		Usage:            usage,
		Warnings:         warnings,
		ProviderMetadata: metadata,
		Responses:        responses,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventEmbedManyFinish, OperationEmbedMany, embedManyResult, map[string]any{
		"usage":             embedManyResult.Usage,
		"value_count":       len(embedManyResult.Values),
		"output.embeddings": embedManyResult.Embeddings,
	})
	return embedManyResult, nil
}

func embeddingChunks(values []string, max int) [][]string {
	if max <= 0 || max >= len(values) {
		return [][]string{values}
	}
	chunks := make([][]string, 0, (len(values)+max-1)/max)
	for start := 0; start < len(values); start += max {
		end := start + max
		if end > len(values) {
			end = len(values)
		}
		chunks = append(chunks, values[start:end])
	}
	return chunks
}

func AddEmbeddingUsage(a, b EmbeddingUsage) EmbeddingUsage {
	return EmbeddingUsage{Tokens: addIntPtr(a.Tokens, b.Tokens)}
}
