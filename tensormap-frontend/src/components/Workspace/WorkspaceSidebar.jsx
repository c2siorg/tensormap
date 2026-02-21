import { useLocation, useNavigate, useParams } from "react-router-dom";
import { Database, BarChart3, Layers, Play, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

const navItems = [
  { label: "Datasets", path: "datasets", icon: Database },
  { label: "Process", path: "process", icon: BarChart3 },
  { label: "Models", path: "models", icon: Layers },
  { label: "Training", path: "training", icon: Play },
];

export default function WorkspaceSidebar({ projectName }) {
  const { projectId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const currentSection = location.pathname.split("/").pop();

  return (
    <div className="flex h-full w-56 flex-col border-r bg-card">
      <div className="p-4">
        <h2 className="truncate text-sm font-semibold">{projectName || "Loading..."}</h2>
      </div>
      <Separator />
      <nav className="flex-1 space-y-1 p-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentSection === item.path;
          return (
            <button
              key={item.path}
              onClick={() => navigate(`/workspace/${projectId}/${item.path}`)}
              className={cn(
                "flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </button>
          );
        })}
      </nav>
      <Separator />
      <div className="p-2">
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start"
          onClick={() => navigate("/projects")}
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          All Projects
        </Button>
      </div>
    </div>
  );
}
