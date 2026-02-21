import { Link } from "react-router-dom";

/**
 * Global top navigation bar displayed on every page.
 *
 * Shows the TensorMap logo and application name, linking back to the
 * projects listing.
 */
export default function AppTopBar() {
  return (
    <header className="flex h-14 items-center border-b bg-card px-4">
      <Link to="/projects" className="flex items-center gap-2 text-foreground no-underline">
        <img src="/logo-black.svg" alt="TensorMap" className="h-7 w-7" />
        <span className="text-lg font-semibold tracking-tight">TensorMap</span>
      </Link>
    </header>
  );
}
