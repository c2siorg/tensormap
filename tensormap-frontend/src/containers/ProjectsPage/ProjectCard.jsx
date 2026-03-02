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
  const [menuOpen, setMenuOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [editName, setEditName] = useState(project.name);
  const [editDescription, setEditDescription] = useState(project.description ?? "");

  const handleCardClick = () => {
    navigate(`/workspace/${project.id}/datasets`);
  };

  const handleMenuToggle = (e) => {
    e.stopPropagation();
    setMenuOpen((prev) => !prev);
  };

  const openDeleteDialog = (e) => {
    e.stopPropagation();
    setMenuOpen(false);
    setDeleteDialogOpen(true);
  };

  const openEditDialog = (e) => {
    e.stopPropagation();
    setMenuOpen(false);
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
      }
    } catch (err) {
      logger.error("Failed to delete project:", err);
    } finally {
      setDeleting(false);
      setDeleteDialogOpen(false);
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
        onUpdated(resp.data.data);
        setEditDialogOpen(false);
      }
    } catch (err) {
      logger.error("Failed to update project:", err);
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
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleMenuToggle}
            aria-label="Project options"
          >
            <MoreVertical className="h-4 w-4" />
          </Button>

          {menuOpen && (
            <div
              className="absolute right-0 top-8 z-20 w-36 rounded-md border bg-popover shadow-md"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                className="flex w-full items-center gap-2 px-3 py-2 text-sm hover:bg-accent"
                onClick={openEditDialog}
              >
                <Pencil className="h-3.5 w-3.5" />
                Edit
              </button>
              <button
                className="flex w-full items-center gap-2 px-3 py-2 text-sm text-destructive hover:bg-accent"
                onClick={openDeleteDialog}
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete
              </button>
            </div>
          )}
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

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent onClick={(e) => e.stopPropagation()}>
          <DialogHeader>
            <DialogTitle>Delete project</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete <strong>{project.name}</strong>? This will permanently
              remove the project along with all its datasets and models. This action cannot be
              undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
              {deleting ? "Deleting..." : "Delete project"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent onClick={(e) => e.stopPropagation()}>
          <DialogHeader>
            <DialogTitle>Edit project</DialogTitle>
            <DialogDescription>Update the project name or description.</DialogDescription>
          </DialogHeader>
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
