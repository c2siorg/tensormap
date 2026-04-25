## Summary

Add ability to export trained TensorFlow models in three popular formats:

- **SavedModel** - TensorFlow's native format for deployment and serving
- **TFLite** - TensorFlow Lite for mobile/edge deployment  
- **ONNX** - Open Neural Network Exchange for cross-platform interoperability

## Changes

### Backend
- New endpoint: GET /api/v1/model/export/{model_name}?format=savedmodel|tflite|onnx
- Service function: export_model_service() handles all three formats
- Includes error handling for missing model, missing file, and unsupported format

### Frontend
- Three export buttons in Training page: Export (SavedModel), Export (TFLite), Export (ONNX)
- exportModel() function in ModelServices.js handles blob download

### Testing
- Unit tests for export_model_service() covering:
  - All three export formats
  - Model not found (404)
  - File not on disk (404)
  - Unsupported format (400)

## Usage

```
GET /api/v1/model/export/my_model?format=savedmodel
GET /api/v1/model/export/my_model?format=tflite  
GET /api/v1/model/export/my_model?format=onnx
```

Note: ONNX export requires tf2onnx package to be installed. If not installed, returns 501 status.