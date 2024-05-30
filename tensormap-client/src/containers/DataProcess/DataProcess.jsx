import React, { useState, useEffect } from "react";
import { Segment, Dropdown, Form, Button, Tab } from "semantic-ui-react";
import * as strings from "../../constants/Strings";
import ModalComponent from "../../components/shared/Modal";
import Metrics from "../../components/Process/Metrics";
import DisplayDataset from "../../components/Process/DisplayDataset";
import SelectFileModal from "../../components/Process/SelectFileModal";
import { getAllFiles, setTargetField, getCovMatrix as getCorrMatrix } from "../../services/FileServices";
import PreprocessData from "../../components/Process/PreprocessData";

function DataProcess() {
    const [state, setState] = useState({
        fileList: null,
        selectedFile: null,
        totalDetails: null,
        showFieldsList: false,
        fieldsList: null,
        targetField: null,
        disableButton: true,
        targetAddedSuccessfully: false,
        modalOpen: false,
        corrMatrix: null,
        dataTypes: null,
        metrics: null,
    });

    const updateFileList = () => {
        getAllFiles().then((response) => {
            const fileList = response.map((file, index) => ({ text: `${file.file_name}.${file.file_type}`, value: file.file_id, key: index }));
            setState((prevState) => ({
                ...prevState,
                fileList,
                totalDetails: response,
            }));
        });
    }

    const panes = [
        {
            menuItem: "Metrics",
            render: () =>
                state.corrMatrix ? (
                    <Tab.Pane style={{ padding: "30px", backgroundColor: "#e6e9f0" }}>
                        <Metrics corrMatrix={state.corrMatrix} dataTypes={state.dataTypes} metrics={state.metrics} />
                    </Tab.Pane>
                ) : (
                    <Tab.Pane style={{ padding: "30px" }}>
                        <SelectFileModal />
                    </Tab.Pane>
                ),
        },
        {
            menuItem: "View Dataset",
            render: () =>
                state.selectedFile !== null ? (
                    <Tab.Pane > 
                        <DisplayDataset fileId={state.selectedFile} />
                    </Tab.Pane>
                ) : (
                    <Tab.Pane style={{ padding: "30px" }}>
                        <SelectFileModal />
                    </Tab.Pane>
                ),
        },
        {
            menuItem: "Preprocess Data",
            render: () =>
                state.selectedFile !== null ? (
                    <Tab.Pane > 
                        <PreprocessData fileId={state.selectedFile} updateFileList = {updateFileList}/>
                    </Tab.Pane>
                ) : (
                    <Tab.Pane style={{ padding: "30px" }}>
                        <SelectFileModal />
                    </Tab.Pane>
                ),
        },
    ];

    
    useEffect(() => {
        getAllFiles().then((response) => {
            const fileList = response.map((file, index) => ({ text: `${file.file_name}.${file.file_type}`, value: file.file_id, key: index }));
            setState((prevState) => ({
                ...prevState,
                fileList,
                totalDetails: response,
            }));
        });
    }, []);
    function enableSubmitButton() {
        if (state.selectedFile !== null && state.targetField !== null) {
            setState({ ...state, disableButton: false });
        }
    }
    function modelClose() {
        setState({ ...state, modalOpen: false });
    }

    function modelOpen() {
        setState({ ...state, modalOpen: true });
    }

    async function fileSelectHandler(event, val) {
        try {
            const response = await getCorrMatrix(val.value);
            const fileDetailsArr = Object.entries(response.data_types).map(([key], index) => ({
                text: key,
                value: index + 1,
                key: index,
            }));

            setState({
                ...state,
                corrMatrix: response.correlation_matrix,
                dataTypes: response.data_types,
                metrics: response.metric,
                selectedFile: val.value,
                showFieldsList: true,
                fieldsList: fileDetailsArr,
            });
        } catch (e) {
            console.error(e);
        }
    }

    function fieldSelectHandler(event, val) {
        setState({ ...state, targetField: val.value }, () => enableSubmitButton());
    }

    const addedSuccessfully = (
        <ModalComponent modalOpen={state.modalOpen} modelClose={modelClose} success Modalmessage={strings.PROCESS_SUCCESS_MODEL_MESSAGE} />
    );

    const errorInAddition = (
        <ModalComponent modalOpen={state.modalOpen} modelClose={modelClose} success={false} Modalmessage={strings.PROCESS_FAIL_MODEL_MESSAGE} />
    );

    let fileFieldsList = (
        <Form.Input fluid size="large" placeholder={strings.PROCESS_SELECT_FILE_FIELD_BEFORE} readOnly style={{ marginTop: "2%" }} />
    );

    if (state.showFieldsList) {
        fileFieldsList = (
            <Dropdown
                style={{ marginTop: "2%" }}
                fluid
                placeholder="Select a Target field"
                search
                selection
                onChange={fieldSelectHandler}
                options={state.fieldsList}
            />
        );
    }

    return (
        <div>
            {state.targetAddedSuccessfully ? addedSuccessfully : errorInAddition}

            <Segment textAlign="center" size="huge">
                {strings.PROCESS_SELECT_FILE_TITLE}
            </Segment>
            <Dropdown fluid placeholder="Files" search selection onChange={fileSelectHandler} options={state.fileList} />
            <Tab panes={panes} style={{ marginTop: "2%" }} />
        </div>
    );
}

export default DataProcess;