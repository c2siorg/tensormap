import { useState } from "react";
import PropTypes from "prop-types";
import { Download, Loader2, CheckCircle2, XCircle } from "lucide-react";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import * as urls from "../../constants/Urls";

const FORMATS = [
  {
    id: "savedmodel",
    label: "SavedModel",
    description: "TensorFlow native format (.zip)",
    ext: "zip",
  },
  {
    id: "tflite",
    label: "TFLite",
    description: "TensorFlow Lite for mobile/edge (.tflite)",
    ext: "tflite",
  },
  {
    id: "onnx",
    label: "ONNX",
    description: "Open Neural Network Exchange (.onnx)",
    ext: "onnx",
  },
];

export default function ModelExport({ modelName, disabled }) {
  const [status, setStatus] = useState({}); // { formatId: "idle"|"loading"|"done"|"error" }
  const [errors, setErrors] = useState({});

  const handleExport = async (formatId) => {
    setStatus((prev) => ({ ...prev, [formatId]: "loading" }));
    setErrors((prev) => ({ ...prev, [formatId]: null }));

    try {
      const apiBase = urls.API_BASE_URL ?? "";
      const url = `${apiBase}/api/model/${encodeURIComponent(modelName)}/export/${formatId}`;
      const resp = await fetch(url);

      if (!resp.ok) {
        const json = await resp.json().catch(() => ({}));
        throw new Error(json.message || `Export failed (${resp.status})`);
      }

      // Trigger browser download
      const blob = await resp.blob();
      const format = FORMATS.find((f) => f.id === formatId);
      const filename = `${modelName}.${format?.ext ?? formatId}`;
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objectUrl);

      setStatus((prev) => ({ ...prev, [formatId]: "done" }));
      setTimeout(() => setStatus((prev) => ({ ...prev, [formatId]: "idle" })), 3000);
    } catch (err) {
      setStatus((prev) => ({ ...prev, [formatId]: "error" }));
      setErrors((prev) => ({ ...prev, [formatId]: err.message }));
    }
  };

  if (!modelName) return null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Export Model</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {FORMATS.map(({ id, label, description }) => {
            const s = status[id] ?? "idle";
            return (
              <div key={id} className="flex flex-col gap-2 rounded-lg border p-3">
                <div>
                  <p className="text-sm font-medium">{label}</p>
                  <p className="text-xs text-muted-foreground">{description}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={disabled || s === "loading"}
                  onClick={() => handleExport(id)}
                  className="mt-auto w-full"
                >
                  {s === "loading" && <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />}
                  {s === "done" && <CheckCircle2 className="mr-2 h-3.5 w-3.5 text-green-500" />}
                  {s === "error" && <XCircle className="mr-2 h-3.5 w-3.5 text-destructive" />}
                  {s === "idle" && <Download className="mr-2 h-3.5 w-3.5" />}
                  {s === "loading"
                    ? "Exporting…"
                    : s === "done"
                      ? "Downloaded"
                      : s === "error"
                        ? "Failed"
                        : "Download"}
                </Button>
                {s === "error" && errors[id] && (
                  <p className="text-xs text-destructive">{errors[id]}</p>
                )}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

ModelExport.propTypes = {
  modelName: PropTypes.string,
  disabled: PropTypes.bool,
};
