import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import CreateProjectDialog from "../CreateProjectDialog";

vi.mock("../../../services/ProjectServices", () => ({
  createProject: vi.fn(),
}));

vi.mock("../../../shared/logger", () => ({
  default: { error: vi.fn() },
}));

import { createProject } from "../../../services/ProjectServices";

const defaultProps = {
  open: true,
  onOpenChange: vi.fn(),
  onCreated: vi.fn(),
};

describe("CreateProjectDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows error when API returns success: false", async () => {
    createProject.mockResolvedValue({
      data: { success: false, message: "Name already taken" },
    });

    render(<CreateProjectDialog {...defaultProps} />);

    const nameInput = screen.getByLabelText("Project Name");
    await userEvent.type(nameInput, "My Project");
    await userEvent.click(screen.getByText("Create Project"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Name already taken");
  });

  it("shows error from err.message on network failure", async () => {
    createProject.mockRejectedValue(new Error("Network Error"));

    render(<CreateProjectDialog {...defaultProps} />);

    const nameInput = screen.getByLabelText("Project Name");
    await userEvent.type(nameInput, "My Project");
    await userEvent.click(screen.getByText("Create Project"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Network Error");
  });

  it("shows error from FastAPI detail array on validation failure", async () => {
    const axiosErr = new Error("Request failed");
    axiosErr.response = {
      data: {
        detail: [
          {
            loc: ["body", "name"],
            msg: "String should have at most 100 characters",
            type: "string_too_long",
          },
        ],
      },
    };
    createProject.mockRejectedValue(axiosErr);

    render(<CreateProjectDialog {...defaultProps} />);

    const nameInput = screen.getByLabelText("Project Name");
    await userEvent.type(nameInput, "My Project");
    await userEvent.click(screen.getByText("Create Project"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("String should have at most 100 characters");
  });

  it("shows generic fallback when error has no readable message", async () => {
    createProject.mockRejectedValue(new Error());

    render(<CreateProjectDialog {...defaultProps} />);

    const nameInput = screen.getByLabelText("Project Name");
    await userEvent.type(nameInput, "My Project");
    await userEvent.click(screen.getByText("Create Project"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Failed to create project");
  });

  it("clears error when dialog is closed", async () => {
    createProject.mockResolvedValue({
      data: { success: false, message: "Name already taken" },
    });

    render(<CreateProjectDialog {...defaultProps} />);

    const nameInput = screen.getByLabelText("Project Name");
    await userEvent.type(nameInput, "My Project");
    await userEvent.click(screen.getByText("Create Project"));

    expect(await screen.findByRole("alert")).toHaveTextContent("Name already taken");

    await userEvent.click(screen.getByText("Cancel"));

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("does not show alert when no error", () => {
    render(<CreateProjectDialog {...defaultProps} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("calls onCreated on success and closes the dialog", async () => {
    const projectData = { id: "1", name: "My Project" };
    createProject.mockResolvedValue({
      data: { success: true, data: projectData },
    });

    const onOpenChange = vi.fn();
    const onCreated = vi.fn();
    render(<CreateProjectDialog open={true} onOpenChange={onOpenChange} onCreated={onCreated} />);

    const nameInput = screen.getByLabelText("Project Name");
    await userEvent.type(nameInput, "My Project");
    await userEvent.click(screen.getByText("Create Project"));

    await waitFor(() => {
      expect(onCreated).toHaveBeenCalledWith(projectData);
      expect(onOpenChange).toHaveBeenCalledWith(false);
    });
  });

  it("disables submit button while loading", async () => {
    createProject.mockReturnValue(new Promise(() => {}));

    render(<CreateProjectDialog {...defaultProps} />);

    const nameInput = screen.getByLabelText("Project Name");
    await userEvent.type(nameInput, "My Project");
    await userEvent.click(screen.getByText("Create Project"));

    expect(screen.getByText("Creating...")).toBeInTheDocument();
    expect(screen.getByText("Creating...")).toBeDisabled();
  });
});
