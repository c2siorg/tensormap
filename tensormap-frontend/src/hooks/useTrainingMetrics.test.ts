import { describe, expect, it } from "vitest";
import {
  mergeEpochMetric,
  mergeEpochMetrics,
  shouldPollTrainingMetrics,
  toEpochMetricUpdate,
} from "./useTrainingMetrics";

describe("mergeEpochMetric", () => {
  it("keeps training values when incoming validation values are null", () => {
    const training = {
      epoch: 1,
      loss: 0.25,
      accuracy: 0.9,
      phase: "train" as const,
      batch: 12,
      steps: 12,
    };
    const merged = mergeEpochMetric([training], {
      epoch: 1,
      loss: null,
      accuracy: null,
      val_loss: 0.3,
      val_accuracy: 0.85,
    });

    expect(merged).toHaveLength(1);
    expect(merged[0]).toMatchObject({
      loss: 0.25,
      accuracy: 0.9,
      val_loss: 0.3,
      val_accuracy: 0.85,
      phase: "train",
    });
  });

  it("normalizes null metric fields to omitted values for a new epoch", () => {
    const merged = mergeEpochMetric([], {
      epoch: 1,
      loss: null,
      accuracy: null,
      val_loss: 0.3,
      val_accuracy: null,
      phase: "train",
    });

    expect(merged).toEqual([{ epoch: 1, val_loss: 0.3, phase: "train" }]);
  });

  it("ignores missing and undefined incoming values, but updates defined metrics", () => {
    const prior = { epoch: 1, loss: 0.25, accuracy: 0.9, val_loss: 0.3, val_accuracy: 0.85 };
    const merged = mergeEpochMetric([prior], {
      epoch: 1,
      loss: 0.2,
      accuracy: undefined,
      val_loss: undefined,
    });

    expect(merged).toEqual([{ ...prior, loss: 0.2 }]);
  });

  it("adds new epochs and updates a repeated epoch without duplication", () => {
    const first = { epoch: 1, loss: 0.25 };
    const epochs = mergeEpochMetric([first], { epoch: 2, loss: 0.4 });
    const updated = mergeEpochMetric(epochs, { epoch: 2, loss: 0.3 });

    expect(updated).toEqual([first, { epoch: 2, loss: 0.3 }]);
  });

  it("preserves legitimate zero values from partial updates", () => {
    const prior = { epoch: 1, loss: 0.25, accuracy: 0.9, val_loss: 0.3 };
    expect(mergeEpochMetric([prior], { epoch: 1, loss: 0, accuracy: 0, val_loss: null })).toEqual([
      { ...prior, loss: 0, accuracy: 0 },
    ]);
  });

  it("merges catch-up history without dropping a newer live epoch", () => {
    const live = [{ epoch: 2, loss: 0.15, val_loss: 0.2 }];
    const catchup = [
      { epoch: 1, loss: 0.4 },
      { epoch: 2, loss: 0.16, val_loss: undefined },
    ];
    expect(mergeEpochMetrics(live, catchup)).toEqual([
      { epoch: 1, loss: 0.4 },
      { epoch: 2, loss: 0.15, val_loss: 0.2 },
    ]);
  });

  it("maps a legacy validation event into validation fields without replacing training values", () => {
    const training = { epoch: 1, loss: 0.2, accuracy: 0.9, batch: 8, steps: 8 };
    const validationUpdate = toEpochMetricUpdate({
      type: "metrics",
      phase: "validation",
      epoch: 1,
      loss: 0,
      accuracy: 0.75,
      batch: 2,
      steps: 2,
    });

    expect(mergeEpochMetric([training], validationUpdate)).toEqual([
      {
        ...training,
        val_loss: 0,
        val_accuracy: 0.75,
        validation_batch: 2,
        validation_steps: 2,
      },
    ]);
  });

  it("adds a complete live epoch record like the catch-up structure", () => {
    expect(
      mergeEpochMetric([], {
        epoch: 2,
        loss: 0.2,
        accuracy: 0.95,
        val_loss: 0.25,
        val_accuracy: 0.9,
      }),
    ).toEqual([{ epoch: 2, loss: 0.2, accuracy: 0.95, val_loss: 0.25, val_accuracy: 0.9 }]);
  });
});

describe("shouldPollTrainingMetrics", () => {
  it("polls a running job while disconnected and stops for connected or terminal jobs", () => {
    expect(shouldPollTrainingMetrics("job-1", false, "running")).toBe(true);
    expect(shouldPollTrainingMetrics("job-1", true, "running")).toBe(false);
    expect(shouldPollTrainingMetrics("job-1", false, "completed")).toBe(false);
    expect(shouldPollTrainingMetrics(null, false, "running")).toBe(false);
  });
});
