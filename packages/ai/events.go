package ai

import "time"

const (
	EventGenerateTextStart      = EventOnStart
	EventGenerateTextStepFinish = EventOnStepFinish
	EventGenerateTextFinish     = EventOnFinish
	EventGenerateTextError      = EventOnError

	EventStreamTextStart      = EventOnStart
	EventStreamTextChunk      = EventOnChunk
	EventStreamTextStepFinish = EventOnStepFinish
	EventStreamTextFinish     = EventOnFinish
	EventStreamTextError      = EventOnError

	EventEmbedStart  = EventOnStart
	EventEmbedFinish = EventOnFinish
	EventEmbedError  = EventOnError

	EventEmbedManyStart  = EventOnStart
	EventEmbedManyFinish = EventOnFinish
	EventEmbedManyError  = EventOnError

	EventGenerateObjectStart  = EventOnStart
	EventGenerateObjectFinish = EventOnFinish
	EventGenerateObjectError  = EventOnError

	EventGenerateImageStart  = EventOnStart
	EventGenerateImageFinish = EventOnFinish
	EventGenerateImageError  = EventOnError

	EventGenerateVideoStart  = EventOnStart
	EventGenerateVideoFinish = EventOnFinish
	EventGenerateVideoError  = EventOnError

	EventGenerateSpeechStart  = EventOnStart
	EventGenerateSpeechFinish = EventOnFinish
	EventGenerateSpeechError  = EventOnError

	EventTranscribeStart  = EventOnStart
	EventTranscribeFinish = EventOnFinish
	EventTranscribeError  = EventOnError

	EventRerankStart  = EventOnStart
	EventRerankFinish = EventOnFinish
	EventRerankError  = EventOnError

	EventUploadFileStart  = EventOnStart
	EventUploadFileFinish = EventOnFinish
	EventUploadFileError  = EventOnError

	EventUploadSkillStart  = EventOnStart
	EventUploadSkillFinish = EventOnFinish
	EventUploadSkillError  = EventOnError

	EventOnStart                  = "onStart"
	EventOnStepStart              = "onStepStart"
	EventOnLanguageModelCallStart = "onLanguageModelCallStart"
	EventOnLanguageModelCallEnd   = "onLanguageModelCallEnd"
	EventOnToolExecutionStart     = "onToolExecutionStart"
	EventOnToolExecutionEnd       = "onToolExecutionEnd"
	EventOnChunk                  = "onChunk"
	EventOnStepFinish             = "onStepFinish"
	EventOnObjectStepStart        = "onObjectStepStart"
	EventOnObjectStepFinish       = "onObjectStepFinish"
	EventOnEmbedStart             = "onEmbedStart"
	EventOnEmbedFinish            = "onEmbedFinish"
	EventOnRerankStart            = "onRerankStart"
	EventOnRerankFinish           = "onRerankFinish"
	EventOnFinish                 = "onFinish"
	EventOnError                  = "onError"

	OperationGenerateText   = "generate_text"
	OperationStreamText     = "stream_text"
	OperationEmbed          = "embed"
	OperationEmbedMany      = "embed_many"
	OperationGenerateObject = "generate_object"
	OperationGenerateImage  = "generate_image"
	OperationGenerateVideo  = "generate_video"
	OperationGenerateSpeech = "generate_speech"
	OperationTranscribe     = "transcribe"
	OperationRerank         = "rerank"
	OperationUploadFile     = "upload_file"
	OperationUploadSkill    = "upload_skill"

	OperationIDGenerateText   = "ai.generateText"
	OperationIDStreamText     = "ai.streamText"
	OperationIDEmbed          = "ai.embed"
	OperationIDEmbedMany      = "ai.embedMany"
	OperationIDGenerateObject = "ai.generateObject"
	OperationIDGenerateImage  = "ai.generateImage"
	OperationIDGenerateVideo  = "ai.generateVideo"
	OperationIDGenerateSpeech = "ai.generateSpeech"
	OperationIDTranscribe     = "ai.transcribe"
	OperationIDRerank         = "ai.rerank"
	OperationIDUploadFile     = "ai.uploadFile"
	OperationIDUploadSkill    = "ai.uploadSkill"
)

type Event struct {
	Name          string
	Operation     string
	OperationID   string
	CallID        string
	StepNumber    *int
	Timestamp     time.Time
	Provider      string
	ModelID       string
	Attributes    map[string]any
	Err           error
	RecordInputs  *bool
	RecordOutputs *bool
	FunctionID    string
}

type StartEvent struct {
	Operation string
	CallID    string
	Provider  string
	ModelID   string
}

type StepFinishEvent struct {
	Operation string
	CallID    string
	Step      *StepResult
}

type FinishEvent struct {
	Operation string
	CallID    string
	Result    any
}

type ChunkEvent struct {
	Operation string
	CallID    string
	Chunk     StreamPart
}

type ErrorEvent struct {
	Operation string
	CallID    string
	Err       error
}

type ToolExecutionStartEvent struct {
	ToolCall ToolCall
	Messages []Message
}

type ToolExecutionEndEvent struct {
	ToolCall ToolCall
	Result   ToolResultPart
	Err      error
}
