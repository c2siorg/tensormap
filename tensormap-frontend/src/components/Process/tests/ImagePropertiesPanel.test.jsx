import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import ImagePropertiesPanel from "../ImagePropertiesPanel";

vi.mock("../../../services/FileServices", () => ({
  getImageProperties: vi.fn(),
  updateImageProperties: vi.fn(),
}));

import { getImageProperties, updateImageProperties } from "../../../services/FileServices";

describe("ImagePropertiesPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("loads image properties, class counts, and updates suggested Input shape interactively", async () => {
    const user = userEvent.setup();
    vi.mocked(getImageProperties).mockResolvedValue({
      file_id: "zip-1",
      file_name: "flowers.zip",
      image_size: 224,
      batch_size: 32,
      color_mode: "rgb",
      label_mode: "int",
      channels: 3,
      suggested_input_shape: "224, 224, 3",
      total_images: 120,
      num_classes: 2,
      classes: [
        { name: "daisy", count: 60 },
        { name: "rose", count: 60 },
      ],
    });
    vi.mocked(updateImageProperties).mockResolvedValue({
      file_id: "zip-1",
      file_name: "flowers.zip",
      image_size: 128,
      batch_size: 16,
      color_mode: "grayscale",
      label_mode: "categorical",
      channels: 1,
      suggested_input_shape: "128, 128, 1",
      total_images: 120,
      num_classes: 2,
      classes: [
        { name: "daisy", count: 60 },
        { name: "rose", count: 60 },
      ],
    });

    render(<ImagePropertiesPanel fileId="zip-1" />);

    await waitFor(() => {
      expect(screen.getByText(/Classes: 2/i)).toBeInTheDocument();
      expect(screen.getByText(/Total Images: 120/i)).toBeInTheDocument();
      expect(screen.getByText("daisy")).toBeInTheDocument();
      expect(screen.getByText("224, 224, 3")).toBeInTheDocument();
    });

    const sizeInput = screen.getByLabelText("Image Size (px)");
    const batchInput = screen.getByLabelText("Default Batch Size");
    const colorSelect = screen.getByLabelText("Color Mode");
    const labelSelect = screen.getByLabelText("Label Mode");

    await user.clear(sizeInput);
    await user.type(sizeInput, "128");
    await user.clear(batchInput);
    await user.type(batchInput, "16");
    await user.selectOptions(colorSelect, "grayscale");
    await user.selectOptions(labelSelect, "categorical");

    expect(screen.getByText("128, 128, 1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Save Image Properties/i }));

    await waitFor(() => {
      expect(updateImageProperties).toHaveBeenCalledWith("zip-1", {
        image_size: 128,
        batch_size: 16,
        color_mode: "grayscale",
        label_mode: "categorical",
      });
      expect(screen.getByText(/Image preprocessing settings saved/i)).toBeInTheDocument();
    });
  });

  it("validates image size bounds and disables save on invalid values", async () => {
    const user = userEvent.setup();
    vi.mocked(getImageProperties).mockResolvedValue({
      file_id: "zip-1",
      file_name: "flowers.zip",
      image_size: 224,
      batch_size: 32,
      color_mode: "rgb",
      label_mode: "int",
      channels: 3,
      suggested_input_shape: "224, 224, 3",
      total_images: 0,
      num_classes: 0,
      classes: [],
    });

    render(<ImagePropertiesPanel fileId="zip-1" />);

    const sizeInput = await screen.findByLabelText("Image Size (px)");
    await user.clear(sizeInput);
    await user.type(sizeInput, "0");

    expect(screen.getByText(/Enter an integer between 1 and 2048/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save Image Properties/i })).toBeDisabled();
  });
});
