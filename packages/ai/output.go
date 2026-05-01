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
	ElementsFromPartial func(partial any, published int) ([]any, int)
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

func ArrayOutput(elementSchema any, options ...OutputOptions) *OutputStrategy {
	opts := firstOutputOptions(options)
	schema := normalizeSchema(elementSchema)
	return &OutputStrategy{
		Name: "array",
		ResponseFormat: &ResponseFormat{
			Type: "json",
			Schema: map[string]any{
				"$schema": "http://json-schema.org/draft-07/schema#",
				"type":    "object",
				"properties": map[string]any{
					"elements": map[string]any{"type": "array", "items": schema},
				},
				"required":             []any{"elements"},
				"additionalProperties": false,
			},
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
			elements, ok := outputElements(value)
			if !ok {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
					Message:      "No object generated: response did not match schema.",
					Cause:        &SDKError{Kind: ErrNoObjectGenerated, Message: "response must be an object with an elements array"},
					Text:         ctx.Text,
					Response:     ctx.Response,
					Usage:        ctx.Usage,
					FinishReason: ctx.FinishReason,
				})
			}
			if err := validateJSONSchema(map[string]any{"type": "array", "items": schema}, elements, "$"); err != nil {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
					Message:      "No object generated: response did not match schema.",
					Cause:        err,
					Text:         ctx.Text,
					Response:     ctx.Response,
					Usage:        ctx.Usage,
					FinishReason: ctx.FinishReason,
				})
			}
			return elements, nil
		},
		ParsePartialOutput: func(text string) (PartialOutputResult, error) {
			if text == "" {
				return PartialOutputResult{}, nil
			}
			value, err := ParsePartialJSON(text)
			if err != nil || value == nil {
				return PartialOutputResult{}, nil
			}
			elements, ok := outputElements(value)
			if !ok {
				return PartialOutputResult{}, nil
			}
			if !completeJSON(text) && len(elements) > 0 {
				elements = elements[:len(elements)-1]
			}
			valid := make([]any, 0, len(elements))
			for _, element := range elements {
				if err := validateJSONSchema(schema, element, "$"); err == nil {
					valid = append(valid, element)
				}
			}
			return PartialOutputResult{Value: valid, OK: true}, nil
		},
		ElementsFromPartial: func(partial any, published int) ([]any, int) {
			elements, ok := partial.([]any)
			if !ok || published >= len(elements) {
				return nil, published
			}
			next := append([]any(nil), elements[published:]...)
			return next, len(elements)
		},
	}
}

func ChoiceOutput(choices []string, options ...OutputOptions) *OutputStrategy {
	opts := firstOutputOptions(options)
	allowed := append([]string(nil), choices...)
	enum := make([]any, 0, len(allowed))
	for _, choice := range allowed {
		enum = append(enum, choice)
	}
	return &OutputStrategy{
		Name: "choice",
		ResponseFormat: &ResponseFormat{
			Type: "json",
			Schema: map[string]any{
				"$schema": "http://json-schema.org/draft-07/schema#",
				"type":    "object",
				"properties": map[string]any{
					"result": map[string]any{"type": "string", "enum": enum},
				},
				"required":             []any{"result"},
				"additionalProperties": false,
			},
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
			choice, ok := outputChoice(value)
			if !ok || !containsString(allowed, choice) {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
					Message:      "No object generated: response did not match schema.",
					Cause:        &SDKError{Kind: ErrNoObjectGenerated, Message: "response must be an object that contains a choice value"},
					Text:         ctx.Text,
					Response:     ctx.Response,
					Usage:        ctx.Usage,
					FinishReason: ctx.FinishReason,
				})
			}
			return choice, nil
		},
		ParsePartialOutput: func(text string) (PartialOutputResult, error) {
			if text == "" {
				return PartialOutputResult{}, nil
			}
			value, err := ParsePartialJSON(text)
			if err != nil || value == nil {
				return PartialOutputResult{}, nil
			}
			choice, ok := outputChoice(value)
			if !ok {
				return PartialOutputResult{}, nil
			}
			matches := prefixMatches(allowed, choice)
			if completeJSON(text) {
				if containsString(matches, choice) {
					return PartialOutputResult{Value: choice, OK: true}, nil
				}
				return PartialOutputResult{}, nil
			}
			if len(matches) == 1 {
				return PartialOutputResult{Value: matches[0], OK: true}, nil
			}
			return PartialOutputResult{}, nil
		},
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

func outputElements(value any) ([]any, bool) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, false
	}
	elements, ok := object["elements"].([]any)
	return elements, ok
}

func outputChoice(value any) (string, bool) {
	object, ok := value.(map[string]any)
	if !ok {
		return "", false
	}
	choice, ok := object["result"].(string)
	return choice, ok
}

func completeJSON(text string) bool {
	_, err := parseObjectText(text)
	return err == nil
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func prefixMatches(values []string, prefix string) []string {
	out := []string{}
	for _, value := range values {
		if len(prefix) <= len(value) && value[:len(prefix)] == prefix {
			out = append(out, value)
		}
	}
	return out
}

func elementsFromPartialOutput(strategy *OutputStrategy, partial any, published int) ([]any, int) {
	strategy = outputStrategyOrDefault(strategy)
	if strategy.ElementsFromPartial == nil {
		return nil, published
	}
	return strategy.ElementsFromPartial(partial, published)
}

func firstOutputOptions(options []OutputOptions) OutputOptions {
	if len(options) == 0 {
		return OutputOptions{}
	}
	return options[0]
}
