import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import FilesList from "../../components/Upload/FilesList/FilesList";
import * as strings from "../../constants/Strings";
import NewFile from "../../components/Upload/NewFile/NewFile";
import { getAllFiles } from "../../services/FileServices";

function DataUpload() {
  const { projectId } = useParams();
  const [fileList, setFileList] = useState(null);
  const [refresh, setRefresh] = useState(false);

  useEffect(() => {
    getAllFiles(projectId)
      .then((response) => {
        const files = response.map((item) => ({
          SavedFileId: item.file_id,
          SavedFileName: item.file_name,
          SavedFileType: item.file_type,
        }));
        setFileList(files);
      })
      .catch((error) => {
        console.error("Error retrieving files:", error);
      });
  }, [refresh, projectId]);

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>{strings.UPLOAD_NEW_FILE_TITLE}</CardTitle>
        </CardHeader>
        <CardContent>
          <NewFile />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>{strings.UPLOAD_FILE_TITLE}</CardTitle>
        </CardHeader>
        <CardContent>
          {fileList ? (
            <FilesList fileList={fileList} setRefresh={setRefresh} refresh={refresh} />
          ) : (
            <div className="flex h-48 items-center justify-center text-muted-foreground">
              {strings.UPLOAD_FILES_LOADING}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default DataUpload;
