package ai

import (
	"encoding/json"
	"fmt"
)

type OutputParseContext struct {
	Text         string
	Response     ResponseMetadata
	Usage        Usage
	FinishReason FinishReason
}

type PartialOutputResult struct {
	Value any
	OK    bool
}

type OutputStrategy struct {
	Name                string
	ResponseFormat      *ResponseFormat
	ParseCompleteOutput func(OutputParseContext) (any, error)
	ParsePartialOutput  func(text string) (PartialOutputResult, error)
}

type OutputOptions struct {
	Schema      any
	Name        string
	Description string
}

type OutputCarrier interface {
	GetOutput() (any, error)
}

func TextOutput() *OutputStrategy {
	return &OutputStrategy{
		Name:           "text",
		ResponseFormat: &ResponseFormat{Type: "text"},
		ParseCompleteOutput: func(ctx OutputParseContext) (any, error) {
			return ctx.Text, nil
		},
		ParsePartialOutput: func(text string) (PartialOutputResult, error) {
			return PartialOutputResult{Value: text, OK: true}, nil
		},
	}
}

func JSONOutput(options ...OutputOptions) *OutputStrategy {
	opts := firstOutputOptions(options)
	return &OutputStrategy{
		Name: "json",
		ResponseFormat: &ResponseFormat{
			Type:        "json",
			Name:        opts.Name,
			Description: opts.Description,
		},
		ParseCompleteOutput: func(ctx OutputParseContext) (any, error) {
			value, err := parseObjectText(ctx.Text)
			if err != nil {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
					Message:      "No object generated: could not parse the response.",
					Cause:        err,
					Text:         ctx.Text,
					Response:     ctx.Response,
					Usage:        ctx.Usage,
					FinishReason: ctx.FinishReason,
				})
			}
			return value, nil
		},
		ParsePartialOutput: parseJSONPartialOutput,
	}
}

func ObjectOutput(schema any, options ...OutputOptions) *OutputStrategy {
	opts := firstOutputOptions(options)
	opts.Schema = schema
	return &OutputStrategy{
		Name: "object",
		ResponseFormat: &ResponseFormat{
			Type:        "json",
			Schema:      normalizeSchema(schema),
			Name:        opts.Name,
			Description: opts.Description,
		},
		ParseCompleteOutput: func(ctx OutputParseContext) (any, error) {
			value, err := parseObjectText(ctx.Text)
			if err != nil {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
					Message:      "No object generated: could not parse the response.",
					Cause:        err,
					Text:         ctx.Text,
					Response:     ctx.Response,
					Usage:        ctx.Usage,
					FinishReason: ctx.FinishReason,
				})
			}
			if _, ok := value.(map[string]any); !ok {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
					Message:      "No object generated: response did not match schema.",
					Cause:        &SDKError{Kind: ErrNoObjectGenerated, Message: "response must be an object"},
					Text:         ctx.Text,
					Response:     ctx.Response,
					Usage:        ctx.Usage,
					FinishReason: ctx.FinishReason,
				})
			}
			if schema != nil {
				if err := validateJSONSchema(normalizeSchema(schema), value, "$"); err != nil {
					return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
						Message:      "No object generated: response did not match schema.",
						Cause:        err,
						Text:         ctx.Text,
						Response:     ctx.Response,
						Usage:        ctx.Usage,
						FinishReason: ctx.FinishReason,
					})
				}
			}
			return value, nil
		},
		ParsePartialOutput: parseJSONPartialOutput,
	}
}

func (r *GenerateTextResult) GetOutput() (any, error) {
	if r == nil {
		return nil, NewNoOutputGeneratedError("", nil)
	}
	if !r.OutputGenerated {
		if r.OutputErr != nil {
			return nil, r.OutputErr
		}
		return nil, NewNoOutputGeneratedError("", nil)
	}
	return r.Output, nil
}

func (r *StreamTextResult) GetOutput() (any, error) {
	if r == nil {
		return nil, NewNoOutputGeneratedError("", nil)
	}
	if !r.OutputGenerated {
		if r.OutputErr != nil {
			return nil, r.OutputErr
		}
		return nil, NewNoOutputGeneratedError("", nil)
	}
	return r.Output, nil
}

func OutputAs[T any](result OutputCarrier) (T, error) {
	var zero T
	value, err := result.GetOutput()
	if err != nil {
		return zero, err
	}
	if typed, ok := value.(T); ok {
		return typed, nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return zero, err
	}
	if err := json.Unmarshal(data, &zero); err != nil {
		return zero, err
	}
	return zero, nil
}

func outputStrategyOrDefault(strategy *OutputStrategy) *OutputStrategy {
	if strategy != nil {
		return strategy
	}
	return TextOutput()
}

func outputResponseFormat(strategy *OutputStrategy, fallback *ResponseFormat) *ResponseFormat {
	if strategy != nil && strategy.ResponseFormat != nil {
		return strategy.ResponseFormat
	}
	return fallback
}

func parseCompleteTextOutput(strategy *OutputStrategy, text string, response ResponseMetadata, usage Usage, finishReason FinishReason) (any, bool, error) {
	if finishReason.Unified != FinishStop {
		return nil, false, NewNoOutputGeneratedError("No output generated.", nil)
	}
	strategy = outputStrategyOrDefault(strategy)
	if strategy.ParseCompleteOutput == nil {
		return nil, false, fmt.Errorf("output strategy %q cannot parse complete output", strategy.Name)
	}
	value, err := strategy.ParseCompleteOutput(OutputParseContext{
		Text:         text,
		Response:     response,
		Usage:        usage,
		FinishReason: finishReason,
	})
	if err != nil {
		return nil, false, err
	}
	return value, true, nil
}

func parsePartialTextOutput(strategy *OutputStrategy, text string) (PartialOutputResult, error) {
	strategy = outputStrategyOrDefault(strategy)
	if strategy.ParsePartialOutput == nil {
		return PartialOutputResult{}, nil
	}
	return strategy.ParsePartialOutput(text)
}

func parseJSONPartialOutput(text string) (PartialOutputResult, error) {
	if text == "" {
		return PartialOutputResult{}, nil
	}
	value, err := ParsePartialJSON(text)
	if err != nil || value == nil {
		return PartialOutputResult{}, nil
	}
	return PartialOutputResult{Value: value, OK: true}, nil
}

func firstOutputOptions(options []OutputOptions) OutputOptions {
	if len(options) == 0 {
		return OutputOptions{}
	}
	return options[0]
}
