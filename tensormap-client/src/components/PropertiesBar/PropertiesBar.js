import React, { useEffect } from 'react';
import {
  Form, Message, Segment, Dropdown,
} from 'semantic-ui-react';
import { getAllFiles } from '../../services/FileServices';

const PropertiesBar = ({ formState, setFormState }) => {
  useEffect(() => {
    // Get the list of datafiles from the backend for the file selector
    getAllFiles()
      .then((response) => {
        const fileList = response.map((file, index) => ({
          text: `${file.file_name}.${file.file_type}`,
          value: file.file_id,
          key: index,
        }));
        setFormState((prevState) => ({
          ...prevState,
          fileList,
          totalDetails: response,
        }));
      })
      .catch((err) => {
        console.error(err);
      });
  }, [setFormState]);

  const fileSelectHandler = (event, val) => {
    const selectedFileDetails = formState.totalDetails.filter((item) => item.file_id === val.value);
    const newFieldsList = selectedFileDetails[0].fields.map((item, index) => ({
      text: item,
      value: item,
      key: index,
    }));
    setFormState((prevState) => ({
      ...prevState,
      selectedFile: val.value,
      fieldsList: newFieldsList,
    }));
  };

  const modalNameHandler = (event) => {
    setFormState((prevState) => ({
      ...prevState,
      modalName: event.target.value,
    }));
  };

  const problemTypeHandler = (event, val) => {
    setFormState((prevState) => ({
      ...prevState,
      problemType: val.value,
    }));
  };

  const optimizerHandler = (event, val) => {
    setFormState((prevState) => ({
      ...prevState,
      optimizer: val.value,
    }));
  };

  const metricHandler = (event, val) => {
    setFormState((prevState) => ({
      ...prevState,
      metric: val.value,
    }));
  };

  const epochCountHandler = (event) => {
    setFormState((prevState) => ({
      ...prevState,
      epochCount: event.target.value,
    }));
  };

  const trainTestRatioHandler = (event) => {
    setFormState((prevState) => ({
      ...prevState,
      trainTestRatio: event.target.value,
    }));
  };

  const batchSizeHandler = (event) => {
    setFormState((prevState) => ({
      ...prevState,
      batchSize: event.target.value,
    }));
  };

  const optimizerOptions = [
    { key: 'opt_1', text: 'Adam', value: 'adam' },
  ];

  const metricOptions = [
    { key: 'acc_1', text: 'Accuracy', value: 'accuracy' },
  ];

  const problemTypeOptions = [
    { key: 'prob_type_1', text: 'Multi class classification [All values float]', value: 1 },
  ];

  const activationOptions = [
    { key: 'act_1', text: 'ReLU', value: 'relu' },
    { key: 'act_2', text: 'Linear', value: 'linear' }
  ];

  const fileFieldsList = (
    <Dropdown
      style={{ marginTop: '2%' }}
      fluid
      placeholder="Target field"
      search
      selection
      onChange={fileSelectHandler}
      options={formState.fieldsList}
    />
  );

  return (
    <div>
      <Segment style={{
        overflow: 'auto', maxHeight: '55vh', minWidth: '15vw', marginLeft: '-13px', marginRight: '-28px',
      }}
      >
        <Message style={{ textAlign: 'center' }}>Code Related </Message>
        <Form>
          <Form.Field required>
            <input
              placeholder="Model Name"
              style={{ marginTop: '5px', marginBottom: '5px' }}
              onChange={modalNameHandler}
            />
          </Form.Field>
          <Form.Field required>
            <Form.Select
              fluid
              options={problemTypeOptions}
              placeholder="Select Problem Type"
              style={{ marginTop: '5px', marginBottom: '5px' }}
              onChange={problemTypeHandler}
            />
          </Form.Field>
          <Form.Field required>
            <Form.Select
              fluid
              options={optimizerOptions}
              placeholder="Optimizer"
              style={{ marginTop: '5px', marginBottom: '5px' }}
              onChange={optimizerHandler}
            />
          </Form.Field>
          <Form.Field required>
            <Form.Select
              fluid
              options={metricOptions}
              placeholder="Result Metrics"
              style={{ marginTop: '5px', marginBottom: '5px' }}
              onChange={metricHandler}
            />
          </Form.Field>
          <Form.Field required>
            <input
              placeholder="No of Epochs"
              type="number"
              style={{ marginTop: '5px', marginBottom: '5px' }}
              onChange={epochCountHandler}
            />
          </Form.Field>
          <Form.Field required>
            <input
              placeholder="Batch Size"
              type="number"
              style={{ marginTop: '5px', marginBottom: '5px' }}
              onChange={batchSizeHandler}
            />
          </Form.Field>
          <Form.Field>
            <Dropdown
              fluid
              placeholder="Select a File"
              search
              selection
              onChange={fileSelectHandler}
              options={formState.fileList}
            />
          </Form.Field>
          <Form.Field required>
            {fileFieldsList}
          </Form.Field>
          <Form.Field required>
            <input
              placeholder="Train:Test ratio"
              type="number"
              style={{ marginTop: '5px', marginBottom: '5px' }}
              onChange={trainTestRatioHandler}
            />
          </Form.Field>
        </Form>
      </Segment>
    </div>
  );
};

export default PropertiesBar;
