package ai

import (
	"context"
	"encoding/json"
	"io"
	"strconv"
	"strings"
)

const (
	OutputObject   = "object"
	OutputArray    = "array"
	OutputEnum     = "enum"
	OutputNoSchema = "no-schema"

	ObjectModeAuto = "auto"
	ObjectModeJSON = "json"
	ObjectModeTool = "tool"
)

func GenerateObject(ctx context.Context, opts GenerateObjectOptions) (objectResult *GenerateObjectResult, err error) {
	emitStart(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnStart, EventGenerateObjectStart, OperationGenerateObject, opts.Model, map[string]any{
		"output": opts.Output,
		"mode":   opts.Mode,
	})
	defer func() {
		if err != nil {
			emitError(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnError, EventGenerateObjectError, OperationGenerateObject, err)
		}
	}()
	responseFormat, err := objectResponseFormat(opts)
	if err != nil {
		return nil, err
	}
	result, err := GenerateText(ctx, GenerateTextOptions{
		Model:                 opts.Model,
		System:                opts.System,
		Prompt:                opts.Prompt,
		Messages:              opts.Messages,
		AllowSystemInMessages: opts.AllowSystemInMessages,
		MaxRetries:            opts.MaxRetries,
		Timeout:               opts.Timeout,
		Headers:               opts.Headers,
		ProviderOptions:       opts.ProviderOptions,
		MaxOutputTokens:       opts.MaxOutputTokens,
		Temperature:           opts.Temperature,
		TopP:                  opts.TopP,
		TopK:                  opts.TopK,
		PresencePenalty:       opts.PresencePenalty,
		FrequencyPenalty:      opts.FrequencyPenalty,
		StopSequences:         opts.StopSequences,
		Seed:                  opts.Seed,
		Reasoning:             opts.Reasoning,
		Download:              opts.Download,
		ResponseFormat:        responseFormat,
		TelemetryOptions:      opts.TelemetryOptions,
	})
	if err != nil {
		return nil, err
	}

	text := objectText(result.Content)
	object, parseErr := parseObjectText(text)
	if parseErr != nil && opts.RepairText != nil {
		repaired, repairErr := opts.RepairText(RepairTextOptions{Text: text, Error: parseErr})
		if repairErr != nil {
			return nil, repairErr
		}
		object, parseErr = parseObjectText(repaired)
		text = repaired
	}
	if parseErr != nil {
		return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
			Message:      "model output is not valid JSON",
			Cause:        parseErr,
			Text:         text,
			Response:     result.Response,
			Usage:        result.Usage,
			FinishReason: FinishReason{Unified: result.FinishReason, Raw: result.RawFinishReason},
		})
	}
	object, err = normalizeAndValidateObjectOutput(opts, object)
	if err != nil {
		return nil, err
	}

	objectResult = &GenerateObjectResult{
		Object:           object,
		FinishReason:     result.FinishReason,
		RawFinishReason:  result.RawFinishReason,
		Usage:            result.Usage,
		Warnings:         result.Warnings,
		ProviderMetadata: result.ProviderMetadata,
		Request:          result.Request,
		Response:         result.Response,
		Reasoning:        reasoningFromParts(result.Content),
		Text:             text,
	}
	emitFinish(ctx, opts.Telemetry, opts.TelemetryOptions, opts.OnFinish, EventGenerateObjectFinish, OperationGenerateObject, objectResult, map[string]any{
		"finish_reason":     objectResult.FinishReason,
		"raw_finish_reason": objectResult.RawFinishReason,
		"usage":             objectResult.Usage,
		"output.object":     objectResult.Object,
	})
	return objectResult, nil
}

func StreamObject(ctx context.Context, opts StreamObjectOptions) (*StreamObjectResult, error) {
	responseFormat, err := objectResponseFormat(opts.GenerateObjectOptions)
	if err != nil {
		return nil, err
	}
	stream, err := StreamText(ctx, StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:                 opts.Model,
			System:                opts.System,
			Prompt:                opts.Prompt,
			Messages:              opts.Messages,
			AllowSystemInMessages: opts.AllowSystemInMessages,
			MaxRetries:            opts.MaxRetries,
			Timeout:               opts.Timeout,
			Headers:               opts.Headers,
			ProviderOptions:       opts.ProviderOptions,
			MaxOutputTokens:       opts.MaxOutputTokens,
			Temperature:           opts.Temperature,
			TopP:                  opts.TopP,
			TopK:                  opts.TopK,
			PresencePenalty:       opts.PresencePenalty,
			FrequencyPenalty:      opts.FrequencyPenalty,
			StopSequences:         opts.StopSequences,
			Seed:                  opts.Seed,
			Reasoning:             opts.Reasoning,
			Download:              opts.Download,
			ResponseFormat:        responseFormat,
		},
		IncludeRawChunks: opts.IncludeRawChunks,
	})
	if err != nil {
		return nil, err
	}

	out := make(chan ObjectStreamPart)
	elements := make(chan any, 16)
	go func() {
		defer close(out)
		defer close(elements)
		var text string
		var lastObject any
		var haveLastObject bool
		var publishedElements int
		emitPartialObject := func(raw any) {
			object, err := ParsePartialJSON(text)
			if err != nil || object == nil {
				return
			}
			if opts.Output == OutputArray && !completeJSON(text) {
				if wrapped, ok := object.(map[string]any); ok {
					if current, ok := wrapped["elements"].([]any); ok && len(current) > 0 {
						wrapped["elements"] = current[:len(current)-1]
					}
				}
			}
			object = normalizePartialObjectOutput(opts.GenerateObjectOptions, object)
			if haveLastObject && DeepEqual(lastObject, object) {
				return
			}
			lastObject = object
			haveLastObject = true
			out <- ObjectStreamPart{Type: "object", Object: object, Raw: raw}
			if opts.Output == OutputArray {
				if newElements, next := newObjectElements(opts.GenerateObjectOptions, object, publishedElements); next != publishedElements {
					publishedElements = next
					for _, element := range newElements {
						sendObjectElement(elements, element)
						out <- ObjectStreamPart{Type: "element", Element: element, Raw: raw}
					}
				}
			}
		}
		for part := range stream.Stream {
			if part.Err != nil {
				out <- ObjectStreamPart{Type: "error", Err: part.Err, Raw: rawStreamPart(opts.IncludeRawChunks, part)}
				continue
			}
			if part.Type == "abort" {
				out <- ObjectStreamPart{Type: "abort", AbortReason: part.AbortReason, Raw: rawStreamPart(opts.IncludeRawChunks, part)}
				continue
			}
			if part.TextDelta != "" {
				text += part.TextDelta
				out <- ObjectStreamPart{Type: "text-delta", TextDelta: part.TextDelta, Raw: rawStreamPart(opts.IncludeRawChunks, part)}
				emitPartialObject(rawStreamPart(opts.IncludeRawChunks, part))
				continue
			}
			if part.ToolInputDelta != "" {
				text += part.ToolInputDelta
				out <- ObjectStreamPart{Type: "text-delta", TextDelta: part.ToolInputDelta, Raw: rawStreamPart(opts.IncludeRawChunks, part)}
				emitPartialObject(rawStreamPart(opts.IncludeRawChunks, part))
				continue
			}
			if part.FinishReason.Unified != "" || part.Usage.TotalTokens != nil {
				if text != "" {
					object, err := parseObjectText(text)
					if err != nil {
						out <- ObjectStreamPart{Type: "error", Err: NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{
							Message:      "model output is not valid JSON",
							Cause:        err,
							Text:         text,
							Response:     stream.Response,
							Usage:        part.Usage,
							FinishReason: part.FinishReason,
						}), Raw: rawStreamPart(opts.IncludeRawChunks, part)}
					} else if _, err := normalizeAndValidateObjectOutput(opts.GenerateObjectOptions, object); err != nil {
						out <- ObjectStreamPart{Type: "error", Err: err, Raw: rawStreamPart(opts.IncludeRawChunks, part)}
					}
				}
				out <- ObjectStreamPart{
					Type:             "finish",
					FinishReason:     part.FinishReason,
					Usage:            part.Usage,
					Warnings:         part.Warnings,
					ProviderMetadata: part.ProviderMetadata,
					Raw:              rawStreamPart(opts.IncludeRawChunks, part),
				}
				continue
			}
			if opts.IncludeRawChunks {
				out <- ObjectStreamPart{Type: "raw", Raw: part}
			}
		}
	}()
	return &StreamObjectResult{Stream: out, Elements: elements, Request: stream.Request, Response: stream.Response}, nil
}

func sendObjectElement(out chan<- any, element any) {
	defer func() {
		_ = recover()
	}()
	select {
	case out <- element:
	default:
		go func() {
			defer func() {
				_ = recover()
			}()
			out <- element
		}()
	}
}

func objectResponseFormat(opts GenerateObjectOptions) (*ResponseFormat, error) {
	output := opts.Output
	if output == "" {
		output = OutputObject
	}
	mode := opts.Mode
	if mode == "" {
		mode = ObjectModeAuto
	}
	if output == OutputNoSchema {
		if opts.Schema != nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Schema is not supported for no-schema output."}
		}
		if opts.SchemaDescription != "" {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Schema description is not supported for no-schema output."}
		}
		if opts.SchemaName != "" {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Schema name is not supported for no-schema output."}
		}
		if opts.Enum != nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Enum values are not supported for no-schema output."}
		}
		if mode != ObjectModeAuto && mode != ObjectModeJSON {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "no-schema output only supports json mode"}
		}
		return &ResponseFormat{Type: "json", Name: opts.SchemaName, Description: opts.SchemaDescription}, nil
	}

	schema := opts.Schema
	switch output {
	case OutputObject:
		if schema == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Schema is required for object output."}
		}
		if opts.Enum != nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Enum values are not supported for object output."}
		}
	case OutputArray:
		if schema == nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Element schema is required for array output."}
		}
		if opts.Enum != nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Enum values are not supported for array output."}
		}
		schema = map[string]any{
			"type": "object",
			"properties": map[string]any{
				"elements": map[string]any{"type": "array", "items": normalizeSchema(schema)},
			},
			"required":             []any{"elements"},
			"additionalProperties": false,
		}
	case OutputEnum:
		if schema != nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Schema is not supported for enum output."}
		}
		if opts.SchemaDescription != "" {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Schema description is not supported for enum output."}
		}
		if opts.SchemaName != "" {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Schema name is not supported for enum output."}
		}
		if len(opts.Enum) == 0 {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Enum values are required for enum output."}
		}
		schema = map[string]any{"type": "string", "enum": append([]string{}, opts.Enum...)}
	default:
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "Invalid output type."}
	}
	return &ResponseFormat{
		Type:        "json",
		Schema:      normalizeSchema(schema),
		Name:        opts.SchemaName,
		Description: opts.SchemaDescription,
	}, nil
}

func objectText(parts []Part) string {
	var out string
	for _, part := range parts {
		switch p := part.(type) {
		case TextPart:
			out += p.Text
		case ToolCallPart:
			if p.ToolName == "json" {
				if p.InputRaw != "" {
					out += p.InputRaw
				} else if p.Input != nil {
					if b, err := json.Marshal(p.Input); err == nil {
						out += string(b)
					}
				}
			}
		}
	}
	return strings.TrimSpace(out)
}

type InjectJSONInstructionOptions struct {
	Prompt       string
	Schema       any
	SchemaPrefix *string
	SchemaSuffix *string
}

func InjectJSONInstruction(opts InjectJSONInstructionOptions) string {
	const defaultSchemaPrefix = "JSON schema:"
	const defaultSchemaSuffix = "You MUST answer with a JSON object that matches the JSON schema above."
	const defaultGenericSuffix = "You MUST answer with JSON."

	lines := []string{}
	if opts.Prompt != "" {
		lines = append(lines, opts.Prompt, "")
	}
	if opts.SchemaPrefix != nil {
		if *opts.SchemaPrefix != "" {
			lines = append(lines, *opts.SchemaPrefix)
		}
	} else if opts.Schema != nil {
		lines = append(lines, defaultSchemaPrefix)
	}
	if opts.Schema != nil {
		if b, err := json.Marshal(normalizeSchema(opts.Schema)); err == nil {
			lines = append(lines, string(b))
		}
	}
	if opts.SchemaSuffix != nil {
		if *opts.SchemaSuffix != "" {
			lines = append(lines, *opts.SchemaSuffix)
		}
	} else if opts.Schema != nil {
		lines = append(lines, defaultSchemaSuffix)
	} else {
		lines = append(lines, defaultGenericSuffix)
	}
	return strings.Join(lines, "\n")
}

func parseObjectText(text string) (any, error) {
	var object any
	decoder := json.NewDecoder(strings.NewReader(strings.TrimSpace(text)))
	decoder.UseNumber()
	if err := decoder.Decode(&object); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return nil, &SDKError{Kind: ErrNoOutputGenerated, Message: "model output contains trailing JSON content", Cause: err}
	}
	return object, nil
}

func normalizeAndValidateObjectOutput(opts GenerateObjectOptions, object any) (any, error) {
	output := opts.Output
	if output == "" {
		output = OutputObject
	}
	switch output {
	case OutputObject:
		if _, ok := object.(map[string]any); !ok {
			return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{Message: "model output is not an object"})
		}
		if opts.Schema != nil {
			if err := validateJSONSchema(normalizeSchema(opts.Schema), object, "$"); err != nil {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{Message: "response did not match schema", Cause: err})
			}
		}
	case OutputArray:
		elements, ok := object.([]any)
		if !ok {
			wrapped, wrappedOK := object.(map[string]any)
			if !wrappedOK {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{Message: "model output is not an array"})
			}
			elements, ok = wrapped["elements"].([]any)
			if !ok {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{Message: "model output is not an array"})
			}
		}
		if opts.Schema != nil {
			schema := map[string]any{"type": "array", "items": normalizeSchema(opts.Schema)}
			if err := validateJSONSchema(schema, elements, "$"); err != nil {
				return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{Message: "response did not match schema", Cause: err})
			}
		}
		return elements, nil
	case OutputEnum:
		value, ok := object.(string)
		if !ok {
			return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{Message: "model output is not an enum string"})
		}
		for _, allowed := range opts.Enum {
			if value == allowed {
				return value, nil
			}
		}
		return nil, NewNoObjectGeneratedError(NoObjectGeneratedErrorOptions{Message: "model output is not in enum"})
	}
	return object, nil
}

func normalizePartialObjectOutput(opts GenerateObjectOptions, object any) any {
	if opts.Output != OutputArray {
		return object
	}
	if elements, ok := object.([]any); ok {
		return elements
	}
	if wrapped, ok := object.(map[string]any); ok {
		if elements, ok := wrapped["elements"].([]any); ok {
			return elements
		}
	}
	return object
}

func newObjectElements(opts GenerateObjectOptions, object any, published int) ([]any, int) {
	elements, ok := object.([]any)
	if !ok || published >= len(elements) {
		return nil, published
	}
	valid := make([]any, 0, len(elements)-published)
	for _, element := range elements[published:] {
		if opts.Schema == nil || validateJSONSchema(normalizeSchema(opts.Schema), element, "$") == nil {
			valid = append(valid, element)
		}
	}
	return valid, len(elements)
}

func validateJSONSchema(schema any, value any, path string) error {
	m, ok := schema.(map[string]any)
	if !ok {
		return nil
	}
	if constValue, ok := m["const"]; ok {
		if !DeepEqual(constValue, value) {
			return &SDKError{Kind: ErrNoObjectGenerated, Message: path + " does not match const"}
		}
	}
	if enumValues, ok := m["enum"].([]any); ok {
		for _, allowed := range enumValues {
			if DeepEqual(allowed, value) {
				return nil
			}
		}
		return &SDKError{Kind: ErrNoObjectGenerated, Message: path + " is not one of the allowed enum values"}
	}
	if enumStrings, ok := m["enum"].([]string); ok {
		for _, allowed := range enumStrings {
			if allowed == value {
				return nil
			}
		}
		return &SDKError{Kind: ErrNoObjectGenerated, Message: path + " is not one of the allowed enum values"}
	}
	if anyOf, ok := schemaSlice(m["anyOf"]); ok {
		if !matchesAnyJSONSchema(anyOf, value, path) {
			return &SDKError{Kind: ErrNoObjectGenerated, Message: path + " does not match anyOf"}
		}
	}
	if oneOf, ok := schemaSlice(m["oneOf"]); ok {
		matches := 0
		for _, candidate := range oneOf {
			if validateJSONSchema(candidate, value, path) == nil {
				matches++
			}
		}
		if matches != 1 {
			return &SDKError{Kind: ErrNoObjectGenerated, Message: path + " does not match exactly one schema"}
		}
	}
	if types := schemaStringSlice(m["type"]); len(types) > 0 {
		if !matchesAnyJSONType(types, value, path) {
			return &SDKError{Kind: ErrNoObjectGenerated, Message: path + " must be one of " + strings.Join(types, ", ")}
		}
	} else if typ, _ := m["type"].(string); typ != "" {
		if err := validateJSONType(typ, value, path); err != nil {
			return err
		}
	}
	switch typed := value.(type) {
	case map[string]any:
		required := schemaStringSlice(m["required"])
		for _, key := range required {
			if _, ok := typed[key]; !ok {
				return &SDKError{Kind: ErrNoObjectGenerated, Message: path + "." + key + " is required"}
			}
		}
		properties, _ := m["properties"].(map[string]any)
		for key, propertySchema := range properties {
			propertyValue, ok := typed[key]
			if !ok {
				continue
			}
			if err := validateJSONSchema(propertySchema, propertyValue, path+"."+key); err != nil {
				return err
			}
		}
		if additional, ok := m["additionalProperties"].(bool); ok && !additional {
			for key := range typed {
				if _, known := properties[key]; !known {
					return &SDKError{Kind: ErrNoObjectGenerated, Message: path + "." + key + " is not allowed"}
				}
			}
		}
	case []any:
		itemSchema, ok := m["items"]
		if !ok {
			return nil
		}
		for i, item := range typed {
			if err := validateJSONSchema(itemSchema, item, path+"["+strconv.Itoa(i)+"]"); err != nil {
				return err
			}
		}
	}
	return nil
}

func matchesAnyJSONSchema(schemas []any, value any, path string) bool {
	for _, schema := range schemas {
		if validateJSONSchema(schema, value, path) == nil {
			return true
		}
	}
	return false
}

func matchesAnyJSONType(types []string, value any, path string) bool {
	for _, typ := range types {
		if validateJSONType(typ, value, path) == nil {
			return true
		}
	}
	return false
}

func schemaSlice(value any) ([]any, bool) {
	switch v := value.(type) {
	case []any:
		return v, true
	case []map[string]any:
		out := make([]any, len(v))
		for i := range v {
			out[i] = v[i]
		}
		return out, true
	default:
		return nil, false
	}
}

func validateJSONType(typ string, value any, path string) error {
	ok := false
	switch typ {
	case "object":
		_, ok = value.(map[string]any)
	case "array":
		_, ok = value.([]any)
	case "string":
		_, ok = value.(string)
	case "number":
		switch value.(type) {
		case float64, json.Number:
			ok = true
		}
	case "integer":
		ok = isJSONInteger(value)
	case "boolean":
		_, ok = value.(bool)
	case "null":
		ok = value == nil
	default:
		ok = true
	}
	if !ok {
		return &SDKError{Kind: ErrNoObjectGenerated, Message: path + " must be " + typ}
	}
	return nil
}

func isJSONInteger(value any) bool {
	switch v := value.(type) {
	case json.Number:
		if _, err := v.Int64(); err == nil {
			return true
		}
		f, err := v.Float64()
		return err == nil && f == float64(int64(f))
	case float64:
		return v == float64(int64(v))
	default:
		return false
	}
}

func schemaStringSlice(value any) []string {
	switch v := value.(type) {
	case []string:
		return append([]string(nil), v...)
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func reasoningFromParts(parts []Part) string {
	var out string
	for _, part := range parts {
		if reasoning, ok := part.(ReasoningPart); ok {
			out += reasoning.Text
		}
	}
	if out == "" {
		return ""
	}
	return out
}

func rawStreamPart(include bool, part StreamPart) any {
	if !include {
		return nil
	}
	return part
}
