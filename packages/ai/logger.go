package ai

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

const FirstWarningInfoMessage = "AI SDK Warning System: To turn off warning logging, set the warning logger to nil."

type WarningLogger interface {
	LogWarnings(warnings []Warning, provider string, modelID string)
}

type WarningLoggerFunc func(warnings []Warning, provider string, modelID string)

func (f WarningLoggerFunc) LogWarnings(warnings []Warning, provider string, modelID string) {
	f(warnings, provider, modelID)
}

var (
	loggerMu      sync.RWMutex
	warningLogger WarningLogger = textWarningLogger{writer: os.Stderr}
	warningFilter WarningFilter
	loggedWarning bool
)

type WarningFilter func(Warning) bool

func SetWarningLogger(logger WarningLogger) {
	loggerMu.Lock()
	defer loggerMu.Unlock()
	if logger == nil {
		warningLogger = WarningLoggerFunc(func([]Warning, string, string) {})
		return
	}
	warningLogger = logger
}

func SetWarningFilter(filter WarningFilter) {
	loggerMu.Lock()
	defer loggerMu.Unlock()
	warningFilter = filter
}

func ResetWarningLoggerState() {
	loggerMu.Lock()
	defer loggerMu.Unlock()
	loggedWarning = false
}

func LogWarnings(warnings []Warning, provider string, modelID string) {
	loggerMu.RLock()
	filter := warningFilter
	logger := warningLogger
	loggerMu.RUnlock()

	filtered := filterWarnings(warnings, filter)
	if len(filtered) == 0 {
		return
	}
	if logger != nil {
		logger.LogWarnings(filtered, provider, modelID)
	}
}

type textWarningLogger struct {
	writer io.Writer
}

func (l textWarningLogger) LogWarnings(warnings []Warning, provider string, modelID string) {
	writer := l.writer
	if writer == nil {
		writer = io.Discard
	}
	loggerMu.Lock()
	if !loggedWarning {
		loggedWarning = true
		_, _ = fmt.Fprintln(writer, FirstWarningInfoMessage)
	}
	loggerMu.Unlock()
	for _, warning := range warnings {
		_, _ = fmt.Fprintln(writer, formatWarningMessage(warning, provider, modelID))
	}
}

func formatWarning(w Warning) string {
	return formatWarningMessage(w, "", "")
}

func formatWarningMessage(w Warning, provider string, modelID string) string {
	scope := ""
	if provider != "" && modelID != "" {
		scope = fmt.Sprintf(" (%s / %s)", provider, modelID)
	}
	prefix := "AI SDK Warning" + scope + ":"
	switch w.Type {
	case "unsupported":
		message := fmt.Sprintf("%s The feature %q is not supported.", prefix, w.Feature)
		if w.Details != "" {
			message += " " + w.Details
		}
		return message
	case "compatibility":
		message := fmt.Sprintf("%s The feature %q is used in a compatibility mode.", prefix, w.Feature)
		if w.Details != "" {
			message += " " + w.Details
		}
		return message
	case "deprecated":
		return fmt.Sprintf("%s Deprecated: %q. %s", prefix, w.Setting, w.Message)
	case "other":
		return fmt.Sprintf("%s %s", prefix, w.Message)
	}
	if b, err := json.MarshalIndent(w, "", "  "); err == nil {
		return fmt.Sprintf("%s %s", prefix, string(b))
	}
	var parts []string
	if w.Type != "" {
		parts = append(parts, "type="+w.Type)
	}
	if w.Feature != "" {
		parts = append(parts, "feature="+w.Feature)
	}
	if w.Setting != "" {
		parts = append(parts, "setting="+w.Setting)
	}
	if w.Message != "" {
		parts = append(parts, "message="+w.Message)
	}
	if w.Details != "" {
		parts = append(parts, "details="+w.Details)
	}
	if len(parts) == 0 {
		return "warning"
	}
	return fmt.Sprintf("%s %s", prefix, strings.Join(parts, " "))
}

func filterWarnings(warnings []Warning, filter WarningFilter) []Warning {
	if len(warnings) == 0 {
		return nil
	}
	if filter == nil {
		return warnings
	}
	filtered := make([]Warning, 0, len(warnings))
	for _, warning := range warnings {
		if filter(warning) {
			filtered = append(filtered, warning)
		}
	}
	return filtered
}

func LogV2CompatibilityWarning(provider string, modelID string) {
	LogWarnings([]Warning{{
		Type:    "compatibility",
		Feature: "specificationVersion",
		Details: "Using v2 specification compatibility mode. Some features may not be available.",
	}}, provider, modelID)
}
