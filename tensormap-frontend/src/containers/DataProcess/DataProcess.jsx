import { useRecoilState } from "recoil";
import { useParams } from "react-router-dom";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import DisplayDataset from "../../components/Process/DisplayDataset";
import SelectFileModal from "../../components/Process/SelectFileModal";
import PreprocessData from "../../components/Process/PreprocessData";
import { getAllFiles } from "../../services/FileServices";
import { projectFiles } from "../../shared/atoms";
import logger from "../../shared/logger";

/**
 * Data preprocessing page. Automatically uses the project's single dataset
 * and renders a two-panel layout (preview + transformations).
 */
function DataProcess() {
  const { projectId } = useParams();
  const [files, setFiles] = useRecoilState(projectFiles);
  const selectedFile = files.length > 0 ? String(files[0].file_id) : null;

  const refreshFiles = () => {
    getAllFiles(projectId)
      .then((response) => setFiles(response))
      .catch((error) => {
        logger.error("Error loading files:", error);
        setFiles([]);
      });
  };

  return (
    <div className="space-y-6">
      {selectedFile === null ? (
        <SelectFileModal />
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Dataset Preview</CardTitle>
            </CardHeader>
            <CardContent>
              <DisplayDataset fileId={selectedFile} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Preprocessing</CardTitle>
            </CardHeader>
            <CardContent>
              <PreprocessData fileId={selectedFile} updateFileList={refreshFiles} />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

export default DataProcess;
