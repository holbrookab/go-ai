package retry

import (
	"context"
	"time"
)

func Do(ctx context.Context, maxRetries int, fn func() error) error {
	if maxRetries < 0 {
		maxRetries = 0
	}
	var last error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := fn(); err != nil {
			last = err
			if attempt == maxRetries {
				return last
			}
			delay := time.Duration(50*(1<<attempt)) * time.Millisecond
			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				timer.Stop()
				return ctx.Err()
			case <-timer.C:
			}
			continue
		}
		return nil
	}
	return last
}
