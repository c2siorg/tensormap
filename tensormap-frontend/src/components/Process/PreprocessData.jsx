import { useState, useEffect } from "react";
import PropTypes from "prop-types";
import { Button } from "@/components/ui/button";
import { getCovMatrix, transformData } from "../../services/FileServices";
import TransformationCreator from "./TransformationCreator";
import TransformationList from "./TransformationList";
import ImagePreprocess from "./ImagePreprocess";
import FeedbackDialog from "../shared/FeedbackDialog";

function PreprocessData({ fileId, fileType, updateFileList }) {
  const [data, setData] = useState(null);
  const [transformations, setTransformations] = useState([]);
  const [disableButton, setDisableButton] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalOpenDuplicateFeature, setModalOpenDuplicateFeature] = useState(false);
  const [datasetCreatedSuccesfully, setDatasetCreatedSuccesfully] = useState(true);
  const [message, setMessage] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (fileType === "zip") return;
        const response = await getCovMatrix(fileId);
        setData(Object.keys(response.data_types));
      } catch (e) {
        console.error(e);
      }
    };
    fetchData();
  }, [fileId, fileType]);

  useEffect(() => {
    if (transformations.length > 0) setDisableButton(false);
    else setDisableButton(true);
  }, [transformations]);

  const handleAddTransformation = (transformation) => {
    if (transformations.some((t) => t.feature === transformation.feature)) {
      setModalOpenDuplicateFeature(true);
    } else setTransformations((prev) => [...prev, transformation]);
  };

  const handleDeleteTransformation = (index) => {
    setTransformations((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async () => {
    const dataTransformed = await transformData(fileId, transformations);
    if (dataTransformed) {
      setDatasetCreatedSuccesfully(true);
      setMessage(dataTransformed.message);
      setModalOpen(true);
    } else {
      setDatasetCreatedSuccesfully(false);
      setMessage(dataTransformed.message);
      setModalOpen(true);
    }
    updateFileList();
  };

  if (fileType !== "zip") {
    return (
      <div>
        <FeedbackDialog
          open={modalOpenDuplicateFeature}
          onClose={() => setModalOpenDuplicateFeature(false)}
          success={false}
          message="Multiple Transformations for the same feature found"
        />
        <FeedbackDialog
          open={modalOpen}
          onClose={() => setModalOpen(false)}
          success={datasetCreatedSuccesfully}
          message={message}
        />
        <TransformationCreator features={data ? data : []} onAdd={handleAddTransformation} />
        {!disableButton ? (
          <TransformationList
            transformations={transformations}
            onDelete={handleDeleteTransformation}
          />
        ) : (
          <></>
        )}
        <Button onClick={handleSubmit} disabled={disableButton} className="mt-4">
          Transform Dataset
        </Button>
      </div>
    );
  } else return <ImagePreprocess fileId={fileId} fileType={fileType} />;
}

PreprocessData.propTypes = {
  fileId: PropTypes.string.isRequired,
  fileType: PropTypes.string.isRequired,
  updateFileList: PropTypes.func.isRequired,
};

export default PreprocessData;
