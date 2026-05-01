package main

import (
	"context"
	"fmt"

	"github.com/holbrookab/go-ai/packages/ai"
	"github.com/holbrookab/go-ai/packages/vertex"
)

func main() {
	provider := vertex.New(vertex.Settings{
		Project:  "my-project",
		Location: "global",
	})

	result, err := ai.GenerateText(context.Background(), ai.GenerateTextOptions{
		Model:  provider.LanguageModel("gemini-2.5-flash"),
		System: "You are concise.",
		Prompt: "Say hello from Go.",
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Text)
}
