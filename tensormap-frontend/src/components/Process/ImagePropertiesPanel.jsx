import { useCallback, useEffect, useState } from "react";
import PropTypes from "prop-types";
import { AlertCircle, CheckCircle2, Image as ImageIcon, Loader2, Save } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getImageProperties, updateImageProperties } from "../../services/FileServices";
import logger from "../../shared/logger";

const COLOR_MODE_CHANNELS = {
  grayscale: 1,
  rgb: 3,
  rgba: 4,
};

/**
 * Displays discovered class folders, image counts, and configurable preprocessing
 * settings (image size, batch size, color mode, label mode) for ZIP image archives.
 */
function ImagePropertiesPanel({ fileId }) {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const [imageSize, setImageSize] = useState("224");
  const [batchSize, setBatchSize] = useState("32");
  const [colorMode, setColorMode] = useState("rgb");
  const [labelMode, setLabelMode] = useState("int");

  const loadProperties = useCallback(async () => {
    if (!fileId) return;
    setLoading(true);
    setError(null);
    setSaveSuccess(false);
    try {
      const data = await getImageProperties(fileId);
      setInfo(data);
      if (data) {
        setImageSize(String(data.image_size ?? 224));
        setBatchSize(String(data.batch_size ?? 32));
        setColorMode(data.color_mode || "rgb");
        setLabelMode(data.label_mode || "int");
      }
    } catch (err) {
      logger.error("Failed to fetch image dataset properties:", err);
      setError(
        err.response?.data?.message ||
          err.response?.data?.detail ||
          "Failed to load image dataset properties.",
      );
    } finally {
      setLoading(false);
    }
  }, [fileId]);

  useEffect(() => {
    loadProperties();
  }, [loadProperties]);

  const parsedImageSize = Number(imageSize);
  const parsedBatchSize = Number(batchSize);
  const isImageSizeValid =
    Number.isInteger(parsedImageSize) && parsedImageSize >= 1 && parsedImageSize <= 2048;
  const isBatchSizeValid =
    Number.isInteger(parsedBatchSize) && parsedBatchSize >= 1 && parsedBatchSize <= 1024;
  const isFormValid = isImageSizeValid && isBatchSizeValid;

  const channels = COLOR_MODE_CHANNELS[colorMode] ?? 3;
  const suggestedShape = isImageSizeValid
    ? `${parsedImageSize}, ${parsedImageSize}, ${channels}`
    : "Enter valid image size";

  const handleSave = async (e) => {
    e.preventDefault();
    if (!isFormValid || saving) return;

    setSaving(true);
    setError(null);
    setSaveSuccess(false);
    try {
      const updated = await updateImageProperties(fileId, {
        image_size: parsedImageSize,
        batch_size: parsedBatchSize,
        color_mode: colorMode,
        label_mode: labelMode,
      });
      setInfo(updated);
      setSaveSuccess(true);
    } catch (err) {
      logger.error("Failed to save image dataset properties:", err);
      setError(
        err.response?.data?.message ||
          err.response?.data?.detail ||
          "Failed to save image dataset properties.",
      );
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex h-32 items-center justify-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading image dataset properties...
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <ImageIcon className="h-4 w-4" />
          Image Dataset Configuration
        </CardTitle>
        <CardDescription>
          Inspect discovered class subfolders and configure image preprocessing parameters used
          during training and code generation.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {saveSuccess && (
          <Alert className="border-emerald-200 bg-emerald-50 text-emerald-900">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            <AlertDescription>
              Image preprocessing settings saved. Use Input layer shape{" "}
              <code className="rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-xs">
                {suggestedShape}
              </code>{" "}
              for convolutional models.
            </AlertDescription>
          </Alert>
        )}

        {info && (
          <div className="space-y-3 rounded-lg border bg-muted/30 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <Badge variant="secondary">Classes: {info.num_classes ?? 0}</Badge>
                <Badge variant="secondary">Total Images: {info.total_images ?? 0}</Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                Suggested Input Shape:{" "}
                <code className="rounded bg-background px-1.5 py-0.5 font-mono font-semibold text-foreground">
                  {suggestedShape}
                </code>
              </div>
            </div>

            {info.classes && info.classes.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {info.classes.map((cls) => (
                  <Badge key={cls.name} variant="outline" className="gap-1.5">
                    <span>{cls.name}</span>
                    <span className="text-muted-foreground">({cls.count})</span>
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                No class subdirectories detected in extracted archive yet. Organize images under one
                subfolder per class (e.g. <code>cats/</code>, <code>dogs/</code>).
              </p>
            )}
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-1.5">
              <Label htmlFor="image-size-input">Image Size (px)</Label>
              <Input
                id="image-size-input"
                type="number"
                min={1}
                max={2048}
                value={imageSize}
                onChange={(e) => {
                  setImageSize(e.target.value);
                  setSaveSuccess(false);
                }}
              />
              {!isImageSizeValid && (
                <p className="text-xs text-red-500">Enter an integer between 1 and 2048.</p>
              )}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="batch-size-input">Default Batch Size</Label>
              <Input
                id="batch-size-input"
                type="number"
                min={1}
                max={1024}
                value={batchSize}
                onChange={(e) => {
                  setBatchSize(e.target.value);
                  setSaveSuccess(false);
                }}
              />
              {!isBatchSizeValid && (
                <p className="text-xs text-red-500">Enter an integer between 1 and 1024.</p>
              )}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="color-mode-select">Color Mode</Label>
              <select
                id="color-mode-select"
                value={colorMode}
                onChange={(e) => {
                  setColorMode(e.target.value);
                  setSaveSuccess(false);
                }}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="rgb">rgb (3 channels)</option>
                <option value="grayscale">grayscale (1 channel)</option>
                <option value="rgba">rgba (4 channels)</option>
              </select>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="label-mode-select">Label Mode</Label>
              <select
                id="label-mode-select"
                value={labelMode}
                onChange={(e) => {
                  setLabelMode(e.target.value);
                  setSaveSuccess(false);
                }}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="int">int (sparse integer labels)</option>
                <option value="categorical">categorical (one-hot vectors)</option>
                <option value="binary">binary (0 / 1 float labels)</option>
              </select>
            </div>
          </div>

          <Button type="submit" disabled={!isFormValid || saving}>
            {saving ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Save className="mr-2 h-4 w-4" />
            )}
            {saving ? "Saving..." : "Save Image Properties"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

ImagePropertiesPanel.propTypes = {
  fileId: PropTypes.string.isRequired,
};

export default ImagePropertiesPanel;
