import React, { useState, useEffect } from "react";
import { Segment, Dropdown, Form, Button, Tab } from "semantic-ui-react";
import { getCovMatrix } from "../../services/FileServices";
import TransformationCreator from "./TransformationCreator";
import TransformationList from "./TransformationList";
import ModalComponent from "../shared/Modal";

function PreprocessData({ fileId }) {
    const [data, setData] = useState(null);
    const [transformations, setTransformations] = useState([]);
    const [disableButton, setDisableButton] = useState(true);
    const [ modalOpen, setModalOpen ] = useState(false);

    function modelClose() {
        setModalOpen(false);
    }

    function modelOpen() {
        setModalOpen(true);
    }

    useEffect(() => {
        console.log(fileId)
        const fetchData = async () => {
            try {
                const response = await getCovMatrix(fileId);
                setData(Object.keys(response.metric));
            } catch (e) {
                console.error(e);
            }
        };
        fetchData();
    }, [fileId]);

    const duplicateFeature = (
        <ModalComponent modalOpen={modalOpen} success={false} modelClose={modelClose}  Modalmessage="Multiple Transformations for the same feature found" />
    ) 

    useEffect(() => {
        if (transformations.length > 0) setDisableButton(false);
        else setDisableButton(true);
    }, [transformations]);

    const handleAddTransformation = (transformation) => {
        if (transformations.some((t) => t.feature === transformation.feature)) {
           modelOpen(); 
        } else setTransformations((prev) => [...prev, transformation]);
    };

    const handleDeleteTransformation = (index) => {
        setTransformations((prev) => prev.filter((_, i) => i !== index));
    };

    const handleSubmit = () => {
        // Send transformations to the backend
    };

    return (
        <div>
        {duplicateFeature}
            <TransformationCreator features={data ? data : []} onAdd={handleAddTransformation} />
            {!disableButton ? <TransformationList transformations={transformations} onDelete={handleDeleteTransformation} /> : <></>}
            <Button color="green" size="large" onClick={handleSubmit} disabled={disableButton}>
                Transform Dataset
            </Button>
        </div>
    );
}

export default PreprocessData;
