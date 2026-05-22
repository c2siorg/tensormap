import { Component } from "react";
import { useNavigate } from "react-router-dom";

/**
 * React Error Boundary that catches rendering errors in its subtree and
 * renders a friendly fallback UI instead of crashing the whole page.
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught an error:", error, info);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return <ErrorFallback error={this.state.error} onRetry={this.handleRetry} />;
    }
    return this.props.children;
  }
}

export function ErrorFallback({ error, onRetry }) {
  const navigate = useNavigate();

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-4 p-8 text-center">
      <h2 className="text-2xl font-semibold text-destructive">Something went wrong</h2>
      <p className="text-muted-foreground max-w-md">
        {error?.message || "An unexpected error occurred. Please try again."}
      </p>
      <div className="flex flex-col sm:flex-row gap-3">
        {onRetry && (
          <button
            className="rounded-md bg-primary px-4 py-2 text-primary-foreground hover:opacity-90"
            onClick={onRetry}
          >
            Try again
          </button>
        )}
        <button
          className="rounded-md border border-border bg-background px-4 py-2 text-foreground hover:bg-accent"
          onClick={() => navigate("/projects")}
        >
          Back to Projects
        </button>
      </div>
    </div>
  );
}

export function PageLoader() {
  return (
    <div className="flex h-screen items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        <span className="text-muted-foreground">Loading page...</span>
      </div>
    </div>
  );
}

export default ErrorBoundary;
