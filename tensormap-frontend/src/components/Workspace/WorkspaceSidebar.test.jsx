import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { RecoilRoot } from "recoil";
import WorkspaceSidebar from "./WorkspaceSidebar";

function LocationEcho() {
  const location = useLocation();
  return <div data-testid="current-path">{location.pathname}</div>;
}

describe("WorkspaceSidebar", () => {
  it("renders all workspace navigation items including Tuning and Analysis & Compare", () => {
    render(
      <RecoilRoot>
        <MemoryRouter initialEntries={["/workspace/proj-1/training"]}>
          <Routes>
            <Route
              path="/workspace/:projectId/*"
              element={<WorkspaceSidebar projectName="Demo Project" />}
            />
          </Routes>
        </MemoryRouter>
      </RecoilRoot>,
    );

    expect(screen.getByText("Demo Project")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Dataset/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Preprocess Data/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Models/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Training/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Tuning/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Analysis & Compare/i })).toBeInTheDocument();
  });

  it("navigates to tuning and analysis routes when clicked", async () => {
    const user = userEvent.setup();

    render(
      <RecoilRoot>
        <MemoryRouter initialEntries={["/workspace/proj-1/training"]}>
          <Routes>
            <Route
              path="/workspace/:projectId/*"
              element={
                <>
                  <WorkspaceSidebar projectName="Demo Project" />
                  <LocationEcho />
                </>
              }
            />
          </Routes>
        </MemoryRouter>
      </RecoilRoot>,
    );

    await user.click(screen.getByRole("button", { name: /Tuning/i }));
    expect(screen.getByTestId("current-path")).toHaveTextContent("/workspace/proj-1/tuning");

    await user.click(screen.getByRole("button", { name: /Analysis & Compare/i }));
    expect(screen.getByTestId("current-path")).toHaveTextContent("/workspace/proj-1/analysis");
  });
});
