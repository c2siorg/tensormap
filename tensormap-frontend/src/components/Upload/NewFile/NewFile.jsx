import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import * as strings from "../../../constants/Strings";
import FeedbackDialog from "../../shared/FeedbackDialog";
import { uploadFile } from "../../../services/FileServices";

function NewFile() {
  const [fileType, setFileType] = useState("");
  const [fileName, setFileName] = useState("");
  const [file, setFile] = useState(null);
  const [fileAddedSuccessfully, setFileAddedSuccessfully] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const isSubmitDisabled = !fileType || !fileName || !file;

  function fileChange(e) {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      setFileName(e.target.files[0].name);
    }
  }

  function handleClose() {
    setModalOpen(false);
    window.location.reload();
  }

  function fileUploadHandler(e) {
    e.preventDefault();
    uploadFile(file)
      .then((success) => {
        setFileAddedSuccessfully(success);
        setModalOpen(true);
      })
      .catch(() => {
        setFileAddedSuccessfully(false);
        setModalOpen(true);
      });
  }

  return (
    <div>
      <FeedbackDialog
        open={modalOpen}
        onClose={handleClose}
        success={fileAddedSuccessfully}
        message={
          fileAddedSuccessfully
            ? strings.UPLOAD_SUCCESS_MODEL_MESSAGE
            : strings.PROCESS_FAIL_MODEL_MESSAGE
        }
      />

      <form onSubmit={fileUploadHandler} className="space-y-4">
        <div className="space-y-2">
          <Label>{strings.UPLOAD_SELECT_FILE_TYPE}</Label>
          <Select onValueChange={(value) => setFileType(value)}>
            <SelectTrigger>
              <SelectValue placeholder={strings.UPLOAD_SELECT_FILE_TYPE} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="csv">CSV</SelectItem>
              <SelectItem value="zip">ZIP</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="file">{strings.UPLOAD_HIDDEN_BUTTON_CONTEXT}</Label>
          <Input id="file" type="file" accept=".csv,.zip" onChange={fileChange} />
          <div className="text-sm text-muted-foreground">
            {strings.UPLOAD_NEW_FILE_INPUT_LABEL}
            {fileName || strings.UPLOAD_NEW_FILE_INPUT_PLACEHOLDER}
          </div>
        </div>

        <Button type="submit" disabled={isSubmitDisabled}>
          {strings.UPLOAD_BUTTON_CONTEXT}
        </Button>
      </form>
    </div>
  );
}

export default NewFile;
