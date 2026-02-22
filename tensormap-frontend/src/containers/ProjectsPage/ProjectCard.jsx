import { useState } from "react";
import { useNavigate } from "react-router-dom";
import PropTypes from "prop-types";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Trash2, Pencil, MoreVertical, FileText, Brain } from "lucide-react";
import { deleteProject, updateProject } from "../../services/ProjectServices";
import logger from "../../shared/logger";

/**
 * Card representing a single ML project.
 *
 * Clicking the card body navigates to the project workspace.
 * The menu exposes Edit and Delete actions. Edit opens an inline
 * dialog pre-filled with the current name/description; Delete shows a
 * confirmation dialog before calling the API.
 *
 * @param {{ project: object, onDeleted: (id: string) => void, onUpdated: (project: object) => void }} props
 */
export default function ProjectCard({ project, onDeleted, onUpdated }) {
  const navigate = useNavigate();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleteError, setDeleteError] = useState(null);
  const [editError, setEditError] = useState(null);
  const [editName, setEditName] = useState(project.name);
  const [editDescription, setEditDescription] = useState(project.description ?? "");

  const handleCardClick = () => {
    navigate(`/workspace/${project.id}/datasets`);
  };

  const openDeleteDialog = (e) => {
    e.stopPropagation();
    setDeleteError(null);
    setDeleteDialogOpen(true);
  };

  const openEditDialog = (e) => {
    e.stopPropagation();
    setEditError(null);
    setEditName(project.name);
    setEditDescription(project.description ?? "");
    setEditDialogOpen(true);
  };

  const handleDelete = async () => {
    setDeleting(true);
    try {
      const resp = await deleteProject(project.id);
      if (resp.data.success) {
        onDeleted(project.id);
        setDeleteDialogOpen(false);
      } else {
        setDeleteError(resp?.data?.message || "Failed to delete project. Please try again.");
      }
    } catch (err) {
      logger.error("Failed to delete project:", err);
      setDeleteError("Failed to delete project. Please try again.");
    } finally {
      setDeleting(false);
    }
  };

  const handleUpdate = async (e) => {
    e.preventDefault();
    if (!editName.trim()) return;
    setSaving(true);
    try {
      const resp = await updateProject(project.id, {
        name: editName.trim(),
        description: editDescription.trim() || null,
      });
      if (resp.data.success) {
        if (resp?.data?.data?.id) {
          onUpdated(resp.data.data);
          setEditDialogOpen(false);
        } else {
          setEditError("Received an unexpected response from the server.");
          logger.error("Unexpected update project response:", resp?.data);
        }
      } else {
        setEditError(resp?.data?.message || "Failed to update project. Please try again.");
      }
    } catch (err) {
      logger.error("Failed to update project:", err);
      setEditError("Failed to update project. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <Card
        className="relative cursor-pointer transition-shadow hover:shadow-md"
        onClick={handleCardClick}
      >
        <div className="absolute right-3 top-3 z-10">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={(e) => e.stopPropagation()}
                aria-label="Project options"
              >
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
              <DropdownMenuItem onClick={openEditDialog}>
                <Pencil className="h-3.5 w-3.5" />
                Edit
              </DropdownMenuItem>
              <DropdownMenuItem className="text-destructive" onClick={openDeleteDialog}>
                <Trash2 className="h-3.5 w-3.5" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <CardHeader className="pr-10">
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

      <Dialog
        open={deleteDialogOpen}
        onOpenChange={(open) => {
          if (!deleting) setDeleteDialogOpen(open);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete project</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete <strong>{project.name}</strong>? This will permanently
              remove the project along with all its datasets and models. This action cannot be
              undone.
            </DialogDescription>
          </DialogHeader>
          {deleteError && (
            <p className="text-sm font-medium text-destructive mt-2">{deleteError}</p>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
              {deleting ? "Deleting..." : "Delete project"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={editDialogOpen}
        onOpenChange={(open) => {
          if (!saving) setEditDialogOpen(open);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit project</DialogTitle>
            <DialogDescription>Update the project name or description.</DialogDescription>
          </DialogHeader>
          {editError && <p className="text-sm font-medium text-destructive mt-2">{editError}</p>}
          <form onSubmit={handleUpdate}>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="edit-name">Project Name</Label>
                <Input
                  id="edit-name"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  maxLength={100}
                  required
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="edit-description">Description (optional)</Label>
                <Textarea
                  id="edit-description"
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                  maxLength={500}
                  rows={3}
                />
              </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setEditDialogOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={saving || !editName.trim()}>
                {saving ? "Saving..." : "Save changes"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}

ProjectCard.propTypes = {
  project: PropTypes.shape({
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    description: PropTypes.string,
    file_count: PropTypes.number,
    model_count: PropTypes.number,
    created_on: PropTypes.string,
    updated_on: PropTypes.string,
  }).isRequired,
  onDeleted: PropTypes.func.isRequired,
  onUpdated: PropTypes.func.isRequired,
};
