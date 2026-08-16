/**
 * Apply Best Banner Component
 * Displays best hyperparameters and allows applying them to the model.
 * @module
 */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Loader2 } from "lucide-react";

interface ApplyBestBannerProps {
  bestParams: {
    optimizer?: string;
    learning_rate?: number;
    batch_size?: number;
    epochs?: number;
  };
  metric: string;
  metricValue: number;
  onApply: () => Promise<void>;
}

export default function ApplyBestBanner({
  bestParams,
  metric,
  metricValue,
  onApply,
}: ApplyBestBannerProps) {
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleApply = async () => {
    setApplying(true);
    setError(null);
    try {
      await onApply();
      setApplied(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to apply parameters");
    } finally {
      setApplying(false);
    }
  };

  return (
    <Card className={applied ? "border-green-500 bg-green-50" : "border-blue-500 bg-blue-50"}>
      <CardContent className="pt-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className={applied ? "w-5 h-5 text-green-600" : "w-5 h-5 text-blue-600"} />
              <h3 className={applied ? "font-semibold text-green-900" : "font-semibold text-blue-900"}>
                {applied ? "✓ Best parameters applied" : "Best Parameters Found"}
              </h3>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              {bestParams.optimizer && (
                <div>
                  <span className="text-xs text-gray-600 uppercase">Optimizer</span>
                  <p className="font-medium capitalize">{bestParams.optimizer}</p>
                </div>
              )}
              {bestParams.learning_rate && (
                <div>
                  <span className="text-xs text-gray-600 uppercase">Learning Rate</span>
                  <p className="font-medium font-mono">
                    {bestParams.learning_rate.toExponential(2)}
                  </p>
                </div>
              )}
              {bestParams.batch_size && (
                <div>
                  <span className="text-xs text-gray-600 uppercase">Batch Size</span>
                  <p className="font-medium">{bestParams.batch_size}</p>
                </div>
              )}
              {bestParams.epochs && (
                <div>
                  <span className="text-xs text-gray-600 uppercase">Epochs</span>
                  <p className="font-medium">{bestParams.epochs}</p>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              <Badge variant="outline" className="border-green-500 text-green-700">
                {metric}: {metricValue.toFixed(4)}
              </Badge>
            </div>

            {error && (
              <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                {error}
              </div>
            )}
          </div>

          <div>
            {!applied && (
              <Button onClick={handleApply} disabled={applying} size="lg">
                {applying ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Applying...
                  </>
                ) : (
                  "Apply Best Parameters"
                )}
              </Button>
            )}
            {applied && (
              <Badge className="bg-green-600 text-white px-4 py-2">
                <CheckCircle2 className="w-4 h-4 mr-1" />
                Applied
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
