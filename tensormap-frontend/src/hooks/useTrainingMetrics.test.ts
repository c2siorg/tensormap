import { describe, expect, it } from "vitest";
import { mergeEpochMetric } from "./useTrainingMetrics";

describe("mergeEpochMetric", () => {
  it("keeps training values when a legacy validation-only event arrives", () => {
    const training = {
      epoch: 1, loss: 0.25, accuracy: 0.9, val_loss: undefined,
      phase: "train" as const, batch: 12, steps: 12,
    };
    const merged = mergeEpochMetric([training], {
      epoch: 1, loss: null as unknown as number, accuracy: null as unknown as number,
      val_loss: 0.3, val_accuracy: 0.85, phase: "validation", batch: 12, steps: 12,
    });

    expect(merged).toHaveLength(1);
    expect(merged[0]).toMatchObject({
      loss: 0.25, accuracy: 0.9, val_loss: 0.3, val_accuracy: 0.85, phase: "train",
    });
  });

  it("adds a complete live epoch record like the catch-up structure", () => {
    expect(mergeEpochMetric([], {
      epoch: 2, loss: 0.2, accuracy: 0.95, val_loss: 0.25, val_accuracy: 0.9,
    })).toEqual([{ epoch: 2, loss: 0.2, accuracy: 0.95, val_loss: 0.25, val_accuracy: 0.9 }]);
  });
});
