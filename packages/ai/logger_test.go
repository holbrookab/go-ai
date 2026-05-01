package ai

import (
	"bytes"
	"strings"
	"testing"
)

func TestLogWarningsUsesConfiguredLogger(t *testing.T) {
	SetWarningFilter(nil)
	var got []Warning
	SetWarningLogger(WarningLoggerFunc(func(warnings []Warning, provider string, modelID string) {
		if provider != "p" || modelID != "m" {
			t.Fatalf("unexpected provider/model: %s %s", provider, modelID)
		}
		got = append(got, warnings...)
	}))
	defer SetWarningLogger(textWarningLogger{writer: ioDiscardForTests{}})
	defer SetWarningFilter(nil)

	LogWarnings([]Warning{{Type: "unsupported", Feature: "seed"}}, "p", "m")
	if len(got) != 1 || got[0].Feature != "seed" {
		t.Fatalf("expected warning to be logged, got %#v", got)
	}
}

func TestTextWarningLoggerAllowsZeroValue(t *testing.T) {
	ResetWarningLoggerState()
	SetWarningLogger(textWarningLogger{})
	defer SetWarningLogger(textWarningLogger{writer: ioDiscardForTests{}})

	LogWarnings([]Warning{{Message: "does not panic"}}, "p", "m")
}

func TestTextWarningLoggerFormatsWarnings(t *testing.T) {
	var buf bytes.Buffer
	ResetWarningLoggerState()
	textWarningLogger{writer: &buf}.LogWarnings([]Warning{{Type: "unsupported", Feature: "seed"}}, "p", "m")
	got := buf.String()
	if !strings.Contains(got, FirstWarningInfoMessage) || !strings.Contains(got, "AI SDK Warning (p / m): The feature \"seed\" is not supported.") {
		t.Fatalf("unexpected warning output: %q", got)
	}
}

func TestLogWarningsFiltersWarnings(t *testing.T) {
	SetWarningFilter(func(w Warning) bool { return w.Type != "compatibility" })
	defer SetWarningFilter(nil)

	var got []Warning
	SetWarningLogger(WarningLoggerFunc(func(warnings []Warning, provider string, modelID string) {
		got = append(got, warnings...)
	}))
	defer SetWarningLogger(textWarningLogger{writer: ioDiscardForTests{}})

	LogWarnings([]Warning{
		{Type: "compatibility", Feature: "specificationVersion"},
		{Type: "other", Message: "keep"},
	}, "p", "m")
	if len(got) != 1 || got[0].Message != "keep" {
		t.Fatalf("expected only non-compatibility warning, got %#v", got)
	}
}

func TestLogV2CompatibilityWarning(t *testing.T) {
	var got []Warning
	SetWarningLogger(WarningLoggerFunc(func(warnings []Warning, provider string, modelID string) {
		if provider != "p" || modelID != "m" {
			t.Fatalf("unexpected provider/model: %s %s", provider, modelID)
		}
		got = append(got, warnings...)
	}))
	defer SetWarningLogger(textWarningLogger{writer: ioDiscardForTests{}})

	LogV2CompatibilityWarning("p", "m")
	if len(got) != 1 || got[0].Type != "compatibility" || got[0].Feature != "specificationVersion" {
		t.Fatalf("unexpected compatibility warning: %#v", got)
	}
}

type ioDiscardForTests struct{}

func (ioDiscardForTests) Write(p []byte) (int, error) {
	return len(p), nil
}
