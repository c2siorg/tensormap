import { useState } from "react";
import PropTypes from "prop-types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { submitImageParameters } from "../../services/FileServices";

const colorModeOptions = [
  { key: "grayscale", value: "grayscale", label: "Grayscale" },
  { key: "rgb", value: "rgb", label: "RGB" },
  { key: "rgba", value: "rgba", label: "RGBA" },
];

const labelModeOptions = [
  { key: "int", value: "int", label: "Int" },
  { key: "categorical", value: "categorical", label: "Categorical" },
  { key: "binary", value: "binary", label: "Binary" },
];

const ImagePreprocess = ({ fileId, fileType }) => {
  const [imageSize, setImageSize] = useState("");
  const [batchSize, setBatchSize] = useState("");
  const [colorMode, setColorMode] = useState("rgb");
  const [labelMode, setLabelMode] = useState("int");

  const handleSubmit = (e) => {
    e.preventDefault();
    const data = {
      fileId: fileId,
      fileType: fileType,
      image_size: parseInt(imageSize, 10),
      batch_size: parseInt(batchSize, 10),
      color_mode: colorMode,
      label_mode: labelMode,
    };
    submitImageParameters(data);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="overflow-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b">
              <th className="border px-3 py-2 text-left font-semibold">Image Size</th>
              <th className="border px-3 py-2 text-left font-semibold">Batch Size</th>
              <th className="border px-3 py-2 text-left font-semibold">Color Mode</th>
              <th className="border px-3 py-2 text-left font-semibold">Label Mode</th>
              <th className="border px-3 py-2 text-left font-semibold">Submit</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b">
              <td className="border px-3 py-2">
                <Input
                  type="number"
                  value={imageSize}
                  onChange={(e) => setImageSize(e.target.value)}
                  required
                />
              </td>
              <td className="border px-3 py-2">
                <Input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value)}
                  required
                />
              </td>
              <td className="border px-3 py-2">
                <Select value={colorMode} onValueChange={(value) => setColorMode(value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {colorModeOptions.map((opt) => (
                      <SelectItem key={opt.key} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </td>
              <td className="border px-3 py-2">
                <Select value={labelMode} onValueChange={(value) => setLabelMode(value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {labelModeOptions.map((opt) => (
                      <SelectItem key={opt.key} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </td>
              <td className="border px-3 py-2 text-center">
                <Button type="submit">Submit</Button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </form>
  );
};

ImagePreprocess.propTypes = {
  fileId: PropTypes.string.isRequired,
  fileType: PropTypes.string.isRequired,
};

export default ImagePreprocess;
