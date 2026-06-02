import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect, vi } from "vitest";
import ErrorBoundary, { ErrorFallback, PageLoader } from "./ErrorBoundary";

describe("ErrorFallback", () => {
  it("renders default message when no error provided", () => {
    render(
      <MemoryRouter>
        <ErrorFallback />
      </MemoryRouter>,
    );
    expect(screen.getByText(/something went wrong/i)).toBeDefined();
  });

  it("renders error message when error is provided", () => {
    render(
      <MemoryRouter>
        <ErrorFallback error={new Error("Test error message")} />
      </MemoryRouter>,
    );
    expect(screen.getByText("Test error message")).toBeDefined();
  });

  it("shows try again button when onRetry is provided", () => {
    render(
      <MemoryRouter>
        <ErrorFallback onRetry={() => {}} />
      </MemoryRouter>,
    );
    expect(screen.getByText("Try again")).toBeDefined();
  });

  it("hides try again button when onRetry is not provided", () => {
    render(
      <MemoryRouter>
        <ErrorFallback />
      </MemoryRouter>,
    );
    expect(screen.queryByText("Try again")).toBeNull();
  });

  it("renders back to projects link", () => {
    render(
      <MemoryRouter>
        <ErrorFallback />
      </MemoryRouter>,
    );
    expect(screen.getByText("Back to Projects")).toBeDefined();
  });
});

describe("PageLoader", () => {
  it("renders loading spinner and text", () => {
    render(<PageLoader />);
    expect(screen.getByText("Loading page...")).toBeDefined();
  });
});

describe("ErrorBoundary", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders children when there is no error", () => {
    render(
      <MemoryRouter>
        <ErrorBoundary>
          <div>Child content</div>
        </ErrorBoundary>
      </MemoryRouter>,
    );
    expect(screen.getByText("Child content")).toBeDefined();
  });

  it("renders error fallback when a child throws", () => {
    const ThrowingComponent = () => {
      throw new Error("Boom!");
    };

    render(
      <MemoryRouter>
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      </MemoryRouter>,
    );

    expect(screen.getByText(/something went wrong/i)).toBeDefined();
    expect(screen.getByText("Boom!")).toBeDefined();
  });

  it("recovers after retry resets state", async () => {
    let shouldThrow = true;
    function FlakyComponent() {
      if (shouldThrow) throw new Error("Boom!");
      return <div>Recovered</div>;
    }

    render(
      <MemoryRouter>
        <ErrorBoundary>
          <FlakyComponent />
        </ErrorBoundary>
      </MemoryRouter>,
    );

    expect(screen.getByText(/something went wrong/i)).toBeDefined();

    shouldThrow = false;
    fireEvent.click(screen.getByText("Try again"));

    expect(screen.getByText("Recovered")).toBeDefined();
  });
});
