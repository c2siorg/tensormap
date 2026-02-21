import { Link } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { Separator } from "@/components/ui/separator";

/**
 * Breadcrumb bar shown above the workspace content area.
 *
 * Displays "Projects > {projectName}" with a link back to the listing.
 *
 * @param {{ projectName: string | undefined }} props
 */
export default function WorkspaceTopBar({ projectName }) {
  return (
    <div>
      <div className="flex h-14 items-center gap-2 px-6">
        <Link to="/projects" className="text-sm text-muted-foreground hover:text-foreground">
          Projects
        </Link>
        <ChevronRight className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">{projectName || "Loading..."}</span>
      </div>
      <Separator />
    </div>
  );
}
