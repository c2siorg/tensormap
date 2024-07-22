import { useState } from "react";
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

  const fileTypes = [
    { key: "csv", text: "CSV", value: "csv" },
    { key: "zip", text: "ZIP", value: "zip" },
  ];
  function submitButtonEnableHandler() {
    setState((prevState) => {
      if (
        prevState.fileType !== "" &&
        prevState.fileName !== "" &&
        prevState.file !== null
      ) {
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
    setState((prevState) => ({
      ...prevState,
      file: e.target.files[0],
      fileName: e.target.files[0].name,
    }));
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
        console.log("error adding file");
        console.error(error);
        setState({ fileAddedSuccessfully: false });
        modelOpen();
      });
  }

  const addedSuccessfully = (
    <ModalComponent
      modalOpen={state.modalOpen}
      modelClose={modelClose}
      success
      Modalmessage={strings.UPLOAD_SUCCESS_MODEL_MESSAGE}
    />
  );

  const errorInAddition = (
    <ModalComponent
      modalOpen={state.modalOpen}
      modelClose={modelClose}
      success={false}
      Modalmessage={strings.PROCESS_FAIL_MODEL_MESSAGE}
    />
  );

  return (
    <div>
      {state.fileAddedSuccessfully ? addedSuccessfully : errorInAddition}

      <Form size="big" onSubmit={fileUploadHandler}>
        <Form.Field required>
          <label htmlFor="fileType">{strings.UPLOAD_SELECT_FILE_TYPE}</label>
          <Form.Select
            fluid
            options={fileTypes}
            onChange={dataTypeHandler}
            placeholder={strings.UPLOAD_SELECT_FILE_TYPE}
            id="fileType"
          />
        </Form.Field>
        <Form.Field>
          <label htmlFor="file">{strings.UPLOAD_HIDDEN_BUTTON_CONTEXT}</label>
          <Button as="label" htmlFor="file" type="button" animated="fade">
            <Button.Content visible>
              <Icon name="file" />
            </Button.Content>
            <Button.Content hidden>
              {strings.UPLOAD_HIDDEN_BUTTON_CONTEXT}
            </Button.Content>
          </Button>
          <input
            type="file"
            id="file"
            accept=".csv, .zip"
            hidden
            onChange={fileChange}
          />

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
          <Button
            color="green"
            size="large"
            type="submit"
            disabled={state.uploadButtonDisabledStatus}
          >
            {strings.UPLOAD_BUTTON_CONTEXT}
          </Button>
        </Form.Field>
      </Form>
    </div>
  );
}

export default NewFile;
