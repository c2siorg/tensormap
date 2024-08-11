import { useState, useEffect } from "react";
import { Button } from "semantic-ui-react";
import { getCovMatrix, transformData } from "../../services/FileServices";
import TransformationCreator from "./TransformationCreator";
import TransformationList from "./TransformationList";
import ImagePreprocess from "./ImagePreprocess";
import ModalComponent from "../shared/Modal";

function PreprocessData({ fileId, fileType, updateFileList }) {
  const [data, setData] = useState(null);
  const [transformations, setTransformations] = useState([]);
  const [disableButton, setDisableButton] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalOpenDuplicateFeature, setModalOpenDuplicateFeature] =
    useState(false);
  const [datasetCreatedSuccesfully, setDatasetCreatedSuccesfully] =
    useState(true);
  const [message, setMessage] = useState("");
  function modelClose() {
    setModalOpen(false);
  }

  function modelOpen() {
    setModalOpen(true);
  }
  function modelCloseDuplicateFeature() {
    setModalOpenDuplicateFeature(false);
  }

  function modelOpenDuplicateFeature() {
    setModalOpenDuplicateFeature(true);
  }

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await getCovMatrix(fileId);
        setData(Object.keys(response.data_types));
      } catch (e) {
        console.error(e);
      }
    };
    fetchData();
  }, [fileId]);

  const duplicateFeature = (
    <ModalComponent
      modalOpen={modalOpenDuplicateFeature}
      success={false}
      modelClose={modelCloseDuplicateFeature}
      Modalmessage="Multiple Transformations for the same feature found"
    />
  );

  const successDatasetCreated = (
    <ModalComponent
      modalOpen={modalOpen}
      success={true}
      modelClose={modelClose}
      Modalmessage={message}
    />
  );
  const errorDatasetCreated = (
    <ModalComponent
      modalOpen={modalOpen}
      success={false}
      modelClose={modelClose}
      Modalmessage={message}
    />
  );

  useEffect(() => {
    if (transformations.length > 0) setDisableButton(false);
    else setDisableButton(true);
  }, [transformations]);

  const handleAddTransformation = (transformation) => {
    if (transformations.some((t) => t.feature === transformation.feature)) {
      modelOpenDuplicateFeature();
    } else setTransformations((prev) => [...prev, transformation]);
  };

  const handleDeleteTransformation = (index) => {
    setTransformations((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async () => {
    const dataTransformed = await transformData(fileId, transformations);
    console.log(dataTransformed);
    if (dataTransformed) {
      setDatasetCreatedSuccesfully(true);
      setMessage(dataTransformed.message);
      modelOpen();
    } else {
      setDatasetCreatedSuccesfully(false);
      setMessage(dataTransformed.message);
      modelOpen();
    }
    updateFileList();
  };
  if (fileType !== "zip") {
    return (
      <div>
        {duplicateFeature}
        {datasetCreatedSuccesfully
          ? successDatasetCreated
          : errorDatasetCreated}
        <TransformationCreator
          features={data ? data : []}
          onAdd={handleAddTransformation}
        />
        {!disableButton ? (
          <TransformationList
            transformations={transformations}
            onDelete={handleDeleteTransformation}
          />
        ) : (
          <></>
        )}
        <Button
          color="green"
          size="large"
          onClick={handleSubmit}
          disabled={disableButton}
        >
          Transform Dataset
        </Button>
      </div>
    );
  } else return <ImagePreprocess fileId={fileId} fileType={fileType} />;
}

export default PreprocessData;
