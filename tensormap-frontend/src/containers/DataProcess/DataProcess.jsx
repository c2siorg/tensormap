import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import * as strings from "../../constants/Strings";
import FeedbackDialog from "../../components/shared/FeedbackDialog";
import Metrics from "../../components/Process/Metrics";
import DisplayDataset from "../../components/Process/DisplayDataset";
import SelectFileModal from "../../components/Process/SelectFileModal";
import { getAllFiles, getCovMatrix as getCorrMatrix } from "../../services/FileServices";
import PreprocessData from "../../components/Process/PreprocessData";

function DataProcess() {
  const { projectId } = useParams();
  const [fileList, setFileList] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileType, setSelectedFileType] = useState(null);
  const [totalDetails, setTotalDetails] = useState(null);
  const [targetAddedSuccessfully, setTargetAddedSuccessfully] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [corrMatrix, setCorrMatrix] = useState(null);
  const [dataTypes, setDataTypes] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const updateFileList = () => {
    getAllFiles(projectId).then((response) => {
      const files = response.map((file, index) => ({
        label: `${file.file_name}.${file.file_type}`,
        type: file.file_type,
        value: String(file.file_id),
        key: index,
      }));
      setFileList(files);
      setTotalDetails(response);
    });
  };

  useEffect(() => {
    getAllFiles(projectId).then((response) => {
      const files = response.map((file, index) => ({
        label: `${file.file_name}.${file.file_type}`,
        type: file.file_type,
        value: String(file.file_id),
        key: index,
      }));
      setFileList(files);
      setTotalDetails(response);
    });
  }, [projectId]);

  async function fileSelectHandler(value) {
    try {
      const fileId = value;
      const fileDetail = totalDetails?.find((f) => f.file_id === fileId);
      const fileType = fileDetail?.file_type;

      if (fileType !== "zip") {
        const response = await getCorrMatrix(fileId);
        setCorrMatrix(response.correlation_matrix);
        setDataTypes(response.data_types);
        setMetrics(response.metric);
      }
      setSelectedFile(fileId);
      setSelectedFileType(fileType);
    } catch (e) {
      console.error(e);
    }
  }

  return (
    <div className="space-y-6">
      <FeedbackDialog
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        success={targetAddedSuccessfully}
        message={
          targetAddedSuccessfully
            ? strings.PROCESS_SUCCESS_MODEL_MESSAGE
            : strings.PROCESS_FAIL_MODEL_MESSAGE
        }
      />

      <Card>
        <CardHeader>
          <CardTitle>{strings.PROCESS_SELECT_FILE_TITLE}</CardTitle>
        </CardHeader>
        <CardContent>
          <Select onValueChange={fileSelectHandler}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a file" />
            </SelectTrigger>
            <SelectContent>
              {fileList?.map((file) => (
                <SelectItem key={file.key} value={file.value}>
                  {file.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      <Tabs defaultValue="metrics">
        <TabsList>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="dataset">View Dataset</TabsTrigger>
          <TabsTrigger value="preprocess">Preprocess Data</TabsTrigger>
        </TabsList>
        <TabsContent value="metrics">
          {corrMatrix ? (
            <Metrics
              corrMatrix={corrMatrix}
              dataTypes={dataTypes}
              fileType={selectedFileType}
              metrics={metrics}
            />
          ) : (
            <SelectFileModal />
          )}
        </TabsContent>
        <TabsContent value="dataset">
          {selectedFile !== null ? (
            <DisplayDataset fileId={selectedFile} fileType={selectedFileType} />
          ) : (
            <SelectFileModal />
          )}
        </TabsContent>
        <TabsContent value="preprocess">
          {selectedFile !== null ? (
            <PreprocessData
              fileId={selectedFile}
              fileType={selectedFileType}
              updateFileList={updateFileList}
            />
          ) : (
            <SelectFileModal />
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default DataProcess;
