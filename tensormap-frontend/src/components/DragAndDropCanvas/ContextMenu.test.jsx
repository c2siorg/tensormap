import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ContextMenu from "./ContextMenu";

const defaultProps = {
  x: 100,
  y: 200,
  onDuplicate: vi.fn(),
  onDelete: vi.fn(),
  onClose: vi.fn(),
};

describe("ContextMenu", () => {
  it("renders Duplicate and Delete buttons", () => {
    render(<ContextMenu {...defaultProps} />);
    expect(screen.getByText("Duplicate")).toBeDefined();
    expect(screen.getByText("Delete")).toBeDefined();
  });

  it("calls onDuplicate when Duplicate is clicked", () => {
    const onDuplicate = vi.fn();
    render(<ContextMenu {...defaultProps} onDuplicate={onDuplicate} />);
    fireEvent.click(screen.getByText("Duplicate"));
    expect(onDuplicate).toHaveBeenCalledTimes(1);
  });

  it("calls onDelete when Delete is clicked", () => {
    const onDelete = vi.fn();
    render(<ContextMenu {...defaultProps} onDelete={onDelete} />);
    fireEvent.click(screen.getByText("Delete"));
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when overlay is clicked", () => {
    const onClose = vi.fn();
    render(<ContextMenu {...defaultProps} onClose={onClose} />);
    const overlay = document.querySelector(".fixed.inset-0");
    fireEvent.click(overlay);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("Delete button has destructive styling", () => {
    render(<ContextMenu {...defaultProps} />);
    const deleteBtn = screen.getByText("Delete");
    expect(deleteBtn.className).toContain("text-destructive");
  });
});
