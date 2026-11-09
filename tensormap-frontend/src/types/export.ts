/**
 * Types for model export functionality.
 */

export type ExportFormat = "savedmodel" | "tflite" | "onnx" | "keras";

export interface FormatInfo {
  available: boolean;
  size_bytes: number | null;
  expires_at: string | null;
  onnx_supported?: boolean;
  onnx_issues?: string[] | null;
}

export interface ExportFormatsResponse {
  formats: {
    savedmodel: FormatInfo;
    tflite: FormatInfo;
    onnx: FormatInfo;
    keras?: FormatInfo;
  };
}

export interface ExportError {
  error: string;
  issues: string[];
  suggestion: string;
}
