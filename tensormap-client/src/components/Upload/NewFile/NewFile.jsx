import React, { useState } from "react";
import { Form, Button, Icon } from "semantic-ui-react";
import * as strings from "../../../constants/Strings";
import ModalComponent from "../../shared/Modal";
import { uploadFile } from "../../../services/FileServices";

function NewFile() {
    const [state, setState] = useState({
        fileType: "",
        fileName: "",
        file: null,
        uploadButtonDisabledStatus: true,
        fileAddedSuccessfully: false,
        modalOpen: false,
    });

    const fileTypes = [{ key: "csv", text: "CSV", value: "csv" }];
    function submitButtonEnableHandler() {
        setState((prevState) => {
            if (prevState.fileType !== "" && prevState.fileName !== "" && prevState.file !== null) {
                return { ...prevState, uploadButtonDisabledStatus: false };
            }
            return prevState;
        });
    }
    function dataTypeHandler(event, val) {
        setState((prevState) => ({ ...prevState, fileType: val.value }));
        submitButtonEnableHandler();
    }

    function fileChange(e) {
        setState((prevState) => ({ ...prevState, file: e.target.files[0], fileName: e.target.files[0].name }));
        submitButtonEnableHandler();
    }
    function modelClose() {
        setState((prevState) => ({ ...prevState, modalOpen: false }));
        window.location.reload();
    }

    function modelOpen() {
        setState((prevState) => ({ ...prevState, modalOpen: true }));
    }
    function fileUploadHandler() {
        uploadFile(state.file)
            .then((fileAddedSuccessfully) => {
                setState({ fileAddedSuccessfully });
                modelOpen();
            })
            .catch((error) => {
                console.log("error adding file")
                console.error(error);
                setState({ fileAddedSuccessfully: false });
                modelOpen();
            });
    }

    const addedSuccessfully = (
        <ModalComponent modalOpen={state.modalOpen} modelClose={modelClose} success Modalmessage={strings.UPLOAD_SUCCESS_MODEL_MESSAGE} />
    );

    const errorInAddition = (
        <ModalComponent modalOpen={state.modalOpen} modelClose={modelClose} success={false} Modalmessage={strings.PROCESS_FAIL_MODEL_MESSAGE} />
    );

    return (
        <div>
            {state.fileAddedSuccessfully ? addedSuccessfully : errorInAddition}

            <Form size="big" onSubmit={fileUploadHandler}>
                <Form.Field required>
                    <label htmlFor="fileType">{strings.UPLOAD_SELECT_FILE_TYPE}</label>
                    <Form.Select fluid options={fileTypes} onChange={dataTypeHandler} placeholder={strings.UPLOAD_SELECT_FILE_TYPE} id="fileType" />
                </Form.Field>
                <Form.Field>
                    <label htmlFor="file">{strings.UPLOAD_HIDDEN_BUTTON_CONTEXT}</label>
                    <Button as="label" htmlFor="file" type="button" animated="fade">
                        <Button.Content visible>
                            <Icon name="file" />
                        </Button.Content>
                        <Button.Content hidden>{strings.UPLOAD_HIDDEN_BUTTON_CONTEXT}</Button.Content>
                    </Button>
                    <input type="file" id="file" accept=".csv" hidden onChange={fileChange} />

                    <Form.Input
                        fluid
                        size="mini"
                        label={strings.UPLOAD_NEW_FILE_INPUT_LABEL}
                        placeholder={strings.UPLOAD_NEW_FILE_INPUT_PLACEHOLDER}
                        readOnly
                        value={state.fileName}
                        id="fileName"
                    />
                </Form.Field>
                <Form.Field>
                    <Button color="green" size="large" type="submit" disabled={state.uploadButtonDisabledStatus}>
                        {strings.UPLOAD_BUTTON_CONTEXT}
                    </Button>
                </Form.Field>
            </Form>
        </div>
    );
}

export default NewFile;
// import React, { Component } from "react";
// import { Form, Button, Icon } from "semantic-ui-react";
// import * as strings from "../../../constants/Strings";
// import ModalComponent from "../../shared/Modal";
// import { uploadFile } from "../../../services/FileServices";

// class NewFile extends Component {
//     constructor(props) {
//         super(props);
//         this.state = {
//             fileType: "",
//             fileName: "",
//             file: null,
//             uploadButtonDisabledStatus: true,
//             fileAddedSuccessfully: false,
//             modalOpen: false,
//         };
//     }

//     componentDidMount() {
//         this.setState((prevstate) => ({
//             ...prevstate,
//             fileType: "",
//             fileName: "",
//             file: null,
//             uploadButtonDisabledStatus: true,
//         }));
//     }

//     fileTypes = [{ key: "csv", text: "CSV", value: "csv" }];

//     dataTypeHandler = (event, val) => {
//         this.setState(
//             (prevstate) => ({ ...prevstate, fileType: val.value }),
//             () => {
//                 this.submitButtonEnableHandler();
//             }
//         );
//     };

//     fileChange = (e) => {
//         this.setState(
//             (prevstate) => ({ ...prevstate, file: e.target.files[0], fileName: e.target.files[0].name }),
//             () => {
//                 this.submitButtonEnableHandler();
//             }
//         );
//     };

//     /*
//      * Submission button enable only after the all the necessary fields added.
//      *
//      * */
//     submitButtonEnableHandler = () => {
//         this.setState((prevState) => {
//             if (prevState.fileType !== "" && prevState.fileName !== "" && prevState.file !== null) {
//                 return { ...prevState, uploadButtonDisabledStatus: false };
//             }
//             return prevState;
//         });
//     };

//     /*
//      * handle file uploads and send a request to backend
//      *
//      * */
//     fileUploadHandler = () => {
//         uploadFile(this.state.file)
//             .then((fileAddedSuccessfully) => {
//                 this.setState({ fileAddedSuccessfully });
//                 this.modelOpen();
//             })
//             .catch((error) => {
//                 console.error(error);
//                 this.setState({ fileAddedSuccessfully: false });
//                 this.modelOpen();
//             });
//     };

//     /*
//      * Model related functions controls the feedback of the request
//      *
//      * */
//     modelClose = () => {
//         this.setState({ ...this.state, modalOpen: false });
//         window.location.reload();
//     };

//     modelOpen = () => this.setState({ modalOpen: true });

//     render() {
//         /*
//          * addedSuccessfully and errorInAddition are modals that will pop up after successful addition or failure.
//          *
//          * */

//         const addedSuccessfully = (
//             <ModalComponent
//                 modalOpen={this.state.modalOpen}
//                 modelClose={this.modelClose}
//                 sucess
//                 Modalmessage={strings.UPLOAD_SUCCESS_MODEL_MESSAGE}
//             />
//         );

//         const errorInAddition = (
//             <ModalComponent
//                 modalOpen={this.state.modalOpen}
//                 modelClose={this.modelClose}
//                 sucess={false}
//                 Modalmessage={strings.PROCESS_FAIL_MODEL_MESSAGE}
//             />
//         );

//         return (
//             <div>
//                 {this.state.fileAddedSuccessfully ? addedSuccessfully : errorInAddition}

//                 <Form size="big" onSubmit={this.fileUploadHandler}>
//                     <Form.Field required>
//                         <label>{strings.UPLOAD_SELECT_FILE_TYPE}</label>
//                         <Form.Select fluid options={this.fileTypes} onChange={this.dataTypeHandler} placeholder={strings.UPLOAD_SELECT_FILE_TYPE} />
//                     </Form.Field>
//                     <Form.Field>
//                         <label>{strings.UPLOAD_HIDDEN_BUTTON_CONTEXT} </label>
//                         <Button as="label" htmlFor="file" type="button" animated="fade">
//                             <Button.Content visible>
//                                 <Icon name="file" />
//                             </Button.Content>
//                             <Button.Content hidden>{strings.UPLOAD_HIDDEN_BUTTON_CONTEXT}</Button.Content>
//                         </Button>
//                         <input type="file" id="file" accept=".csv" hidden onChange={this.fileChange} />

//                         <Form.Input
//                             fluid
//                             size="mini"
//                             label={strings.UPLOAD_NEW_FILE_INPUT_LABEL}
//                             placeholder={strings.UPLOAD_NEW_FILE_INPUT_PLACEHOLDER}
//                             readOnly
//                             value={this.state.fileName}
//                         />
//                     </Form.Field>
//                     <Form.Field>
//                         <Button color="green" size="large" type="submit" disabled={this.state.uploadButtonDisabledStatus}>
//                             {strings.UPLOAD_BUTTON_CONTEXT}
//                         </Button>
//                     </Form.Field>
//                 </Form>
//             </div>
//         );
//     }
// }

// export default NewFile;
