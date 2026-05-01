package main

import (
	"context"
	"fmt"

	"github.com/holbrookab/go-ai/packages/ai"
	"github.com/holbrookab/go-ai/packages/bedrock"
)

func main() {
	provider := bedrock.New(bedrock.Settings{
		Region: "us-east-1",
	})

	result, err := ai.GenerateText(context.Background(), ai.GenerateTextOptions{
		Model:  provider.LanguageModel("anthropic.claude-3-haiku-20240307-v1:0"),
		System: "You are concise.",
		Prompt: "Say hello from Go.",
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Text)
}
