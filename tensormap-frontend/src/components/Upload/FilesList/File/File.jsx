import PropTypes from "prop-types";
import { Button } from "@/components/ui/button";
import { deleteFile } from "../../../../services/FileServices";

function File(props) {
  const deleteFileHandler = async () => {
    await deleteFile(props.savedFileId);
    props.setRefresh(!props.refresh);
  };

  return (
    <div className="mb-2 flex items-center justify-between rounded-md border p-3">
      <div className="flex gap-4">
        <span className="font-medium">{props.savedFileName}</span>
        <span className="text-muted-foreground">{props.savedFileType}</span>
      </div>
      <Button variant="destructive" size="sm" onClick={deleteFileHandler}>
        Delete Dataset
      </Button>
    </div>
  );
}

File.propTypes = {
  savedFileName: PropTypes.string.isRequired,
  savedFileType: PropTypes.string.isRequired,
  savedFileId: PropTypes.string.isRequired,
  setRefresh: PropTypes.func.isRequired,
  refresh: PropTypes.bool.isRequired,
};

export default File;
