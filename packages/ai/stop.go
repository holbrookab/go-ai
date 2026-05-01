package ai

import "context"

type StopCondition func(context.Context, []*StepResult) (bool, error)

func StepCount(n int) StopCondition {
	return func(_ context.Context, steps []*StepResult) (bool, error) {
		return len(steps) >= n, nil
	}
}

func LoopFinished() StopCondition {
	return func(context.Context, []*StepResult) (bool, error) { return false, nil }
}

func HasToolCall(names ...string) StopCondition {
	nameSet := map[string]struct{}{}
	for _, name := range names {
		nameSet[name] = struct{}{}
	}
	return func(_ context.Context, steps []*StepResult) (bool, error) {
		if len(steps) == 0 {
			return false, nil
		}
		for _, call := range steps[len(steps)-1].ToolCalls {
			if _, ok := nameSet[call.ToolName]; ok {
				return true, nil
			}
		}
		return false, nil
	}
}

func stopConditionMet(ctx context.Context, conditions []StopCondition, steps []*StepResult) (bool, error) {
	for _, condition := range conditions {
		ok, err := condition(ctx, steps)
		if err != nil || ok {
			return ok, err
		}
	}
	return false, nil
}
