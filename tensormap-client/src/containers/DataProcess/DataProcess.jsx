import React, { useState, useEffect } from "react";
import { Segment, Dropdown, Form, Button, Tab } from "semantic-ui-react";
import * as strings from "../../constants/Strings";
import ModalComponent from "../../components/shared/Modal";
import Metrics from "../../components/Process/Metrics";
import SelectFileModal from "../../components/Process/SelectFileModal";
import { getAllFiles, setTargetField, getCovMatrix as getCorrMatrix } from "../../services/FileServices";

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
        { menuItem: "View Dataset", render: () => <Tab.Pane loading>Tab 2 Content</Tab.Pane> },
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
    function setTargetFieldHandler() {
        const { selectedFile, targetField } = state;
        setTargetField(selectedFile, targetField)
            .then((targetAddedSuccessfully) => {
                setState({ targetAddedSuccessfully });
                modelOpen();
            })
            .catch((error) => {
                console.error(error);
                setState({ targetAddedSuccessfully: false });
                modelOpen();
            });
    }

    function fileSelectHandler(event, val) {
        getCorrMatrix(val.value).then((response) => {
            setState({
                ...state,
                corrMatrix: response.correlation_matrix,
                dataTypes: response.data_types,
                metrics: response.metric,
            });
        });
        const promise = new Promise((resolve) => {
            setState({ ...state, selectedFile: val.value, showFieldsList: true }, () => resolve());
        });

        promise.then(() => {
            const selectedFIleDetails = state.totalDetails.filter((item) => item.file_id === state.selectedFile);

            setState({
                ...state,
                fieldsList: selectedFIleDetails[0].fields.map((item, index) => ({ text: item, value: item, key: index })),
            });
        });
    }

    function fieldSelectHandler(event, val) {
        setState({ ...state, targetField: val.value }, () => enableSubmitButton());
    }

    const addedSuccessfully = (
        <ModalComponent modalOpen={state.modalOpen} modelClose={modelClose} sucess Modalmessage={strings.PROCESS_SUCCESS_MODEL_MESSAGE} />
    );

    const errorInAddition = (
        <ModalComponent modalOpen={state.modalOpen} modelClose={modelClose} sucess={false} Modalmessage={strings.PROCESS_FAIL_MODEL_MESSAGE} />
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
            {fileFieldsList}
            <Button color="green" size="large" style={{ marginTop: "2%" }} onClick={setTargetFieldHandler} disabled={state.disableButton}>
                {strings.PROCESS_TARGET_FIELD_SUBMIT_BUTTON}
            </Button>
            <Tab panes={panes} style={{ marginTop: "2%" }} />
        </div>
    );
}

export default DataProcess;
// import React, { Component } from 'react';
// import {
//   Segment, Dropdown, Form, Button, Tab,
// } from 'semantic-ui-react';
// import * as strings from '../../constants/Strings';
// import ModalComponent from '../../components/shared/Modal';
// import Metrics from '../../components/Process/Metrics';
// import SelectFileModal from '../../components/Process/SelectFileModal';
// import { getAllFiles, setTargetField, getCovMatrix as getCorrMatrix } from '../../services/FileServices';

// class DataProcess extends Component {
//   state = {
//     fileList: null,
//     selectedFile: null,
//     totalDetails: null,
//     showFieldsList: false,
//     fieldsList: null,
//     targetField: null,
//     disableButton: true,
//     targetAddedSuccessfully: false,
//     modalOpen: false,
//     corrMatrix: null,
//     dataTypes: null,
//     metrics: null,
//   };

//   panes = [
//     {
//       menuItem: 'Metrics',
//       render: () => (this.state.corrMatrix ? <Tab.Pane style={{ padding: '30px', backgroundColor: '#e6e9f0' }}><Metrics corrMatrix={this.state.corrMatrix} dataTypes={this.state.dataTypes} metrics={this.state.metrics} /></Tab.Pane> : <Tab.Pane style={{ padding: '30px' }}><SelectFileModal /></Tab.Pane>),
//     },
//     { menuItem: 'View Dataset', render: () => <Tab.Pane loading>Tab 2 Content</Tab.Pane> },
//   ];

//   componentDidMount() {
//     /*
//         * In this react life cycle hook, file data is fetch from backend and added to state
//         *
//         * */
//     getAllFiles()
//       .then(
//         (response) => {
//           const fileList = response.map((file, index) => (
//             { text: `${file.file_name}.${file.file_type}`, value: file.file_id, key: index }
//           ));
//           this.setState((prevState) => ({
//             ...prevState,
//             fileList,
//             totalDetails: response,
//           }));
//         },
//       );
//   }

//   fileSelectHandler = (event, val) => {
//     getCorrMatrix(val.value)
//       .then(
//         (response) => {
//           console.log(response);
//           this.setState({
//             ...this.state, corrMatrix: response.correlation_matrix, dataTypes: response.data_types, metrics: response.metric,
//           });
//         },

//       );
//     const promise = new Promise((resolve) => {
//       this.setState({ ...this.state, selectedFile: val.value, showFieldsList: true }, () => resolve());
//     });

//     promise.then(() => {
//       const selectedFIleDetails = this.state.totalDetails.filter((item) => item.file_id === this.state.selectedFile);

//       this.setState({
//         ...this.state,
//         fieldsList: selectedFIleDetails[0].fields.map(
//           (item, index) => ({ text: item, value: item, key: index }),
//         ),
//       });
//     });
//   };

//   fieldSelectHandler = (event, val) => {
//     this.setState({ ...this.state, targetField: val.value }, () => this.enableSubmitButton());
//   };

//   /*
//     * Submission button enable only after the all the necessary fields added.
//     *
//     * */
//   enableSubmitButton = () => {
//     if (this.state.selectedFile !== null && this.state.targetField !== null) {
//       this.setState({ ...this.state, disableButton: false });
//     }
//   };

//   // Select the target field for a particular data file
//   setTargetFieldHandler = () => {
//     const { selectedFile, targetField } = this.state;
//     setTargetField(selectedFile, targetField)
//       .then((targetAddedSuccessfully) => {
//         this.setState({ targetAddedSuccessfully });
//         this.modelOpen();
//       })
//       .catch((error) => {
//         console.error(error);
//         this.setState({ targetAddedSuccessfully: false });
//         this.modelOpen();
//       });
//   };

//   /*
//     * Model related functions controls the feedback of the request
//     *
//     * */
//   modelClose = () => {
//     this.setState({ ...this.state, modalOpen: false });
//   };

//   modelOpen = () => this.setState({ ...this.state, modalOpen: true });

//   render() {
//     /*
//         * addedSuccessfully and errorInAddition are modals that will pop up after successful addition or failure.
//         *
//         * */
//     const addedSuccessfully = (
//       <ModalComponent
//         modalOpen={this.state.modalOpen}
//         modelClose={this.modelClose}
//         sucess
//         Modalmessage={strings.PROCESS_SUCCESS_MODEL_MESSAGE}
//       />
//     );

//     const errorInAddition = (
//       <ModalComponent
//         modalOpen={this.state.modalOpen}
//         modelClose={this.modelClose}
//         sucess={false}
//         Modalmessage={strings.PROCESS_FAIL_MODEL_MESSAGE}
//       />
//     );

//     /*
//         * Until a file is selected field list of a file is not showing
//         *
//         * */
//     let fileFieldsList = (
//       <Form.Input
//         fluid
//         size="large"
//         placeholder={strings.PROCESS_SELECT_FILE_FIELD_BEFORE}
//         readOnly
//         style={{ marginTop: '2%' }}
//       />
//     );

//     if (this.state.showFieldsList) {
//       fileFieldsList = (
//         <Dropdown
//           style={{ marginTop: '2%' }}
//           fluid
//           placeholder="Select a Target field"
//           search
//           selection
//           onChange={this.fieldSelectHandler}
//           options={this.state.fieldsList}
//         />
//       );
//     }

//     return (
//       <div>

//         {(this.state.targetAddedSuccessfully) ? addedSuccessfully : errorInAddition}

//         <Segment
//           textAlign="center"
//           size="huge"
//         >
//           {strings.PROCESS_SELECT_FILE_TITLE}
//         </Segment>
//         <Dropdown
//           fluid
//           placeholder="Files"
//           search
//           selection
//           onChange={this.fileSelectHandler}
//           options={this.state.fileList}
//         />
//         {fileFieldsList}
//         <Button
//           color="green"
//           size="large"
//           style={{ marginTop: '2%' }}
//           onClick={this.setTargetFieldHandler}
//           disabled={this.state.disableButton}
//         >
//           {strings.PROCESS_TARGET_FIELD_SUBMIT_BUTTON}
//         </Button>
//         <Tab panes={this.panes} style={{ marginTop: '2%' }} />
//       </div>
//     );
//   }
// }

// export default DataProcess;