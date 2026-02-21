import { useNavigate } from "react-router-dom";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FileText, Brain } from "lucide-react";

export default function ProjectCard({ project }) {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(`/workspace/${project.id}/datasets`);
  };

  return (
    <Card className="cursor-pointer transition-shadow hover:shadow-md" onClick={handleClick}>
      <CardHeader>
        <CardTitle className="text-lg">{project.name}</CardTitle>
        {project.description && (
          <CardDescription className="line-clamp-2">{project.description}</CardDescription>
        )}
      </CardHeader>
      <CardContent>
        <div className="flex gap-3">
          <Badge variant="secondary" className="flex items-center gap-1">
            <FileText className="h-3 w-3" />
            {project.file_count ?? 0} files
          </Badge>
          <Badge variant="secondary" className="flex items-center gap-1">
            <Brain className="h-3 w-3" />
            {project.model_count ?? 0} models
          </Badge>
        </div>
      </CardContent>
      <CardFooter className="text-xs text-muted-foreground">
        {project.updated_on
          ? `Updated ${new Date(project.updated_on).toLocaleDateString()}`
          : `Created ${new Date(project.created_on).toLocaleDateString()}`}
      </CardFooter>
    </Card>
  );
}
