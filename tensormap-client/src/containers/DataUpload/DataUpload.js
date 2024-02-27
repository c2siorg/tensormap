import React, { useState, useEffect } from 'react';
import {
  Grid, Segment, Divider, Dimmer, Loader,
} from 'semantic-ui-react';
import FilesList from '../../components/Upload/FilesList/FilesList';
import * as strings from '../../constants/Strings';
import NewFile from '../../components/Upload/NewFile/NewFile';
import { getAllFiles } from '../../services/FileServices';

function DataUpload() {
  const [fileList, setFileList] = useState(null);

  useEffect(() => {
    getAllFiles()
      .then((response) => {
        const updatedFileList = response.map((item) => ({
          SavedFileName: item.file_name,
          SavedFileType: item.file_type,
        }));
        setFileList(updatedFileList);
      })
      .catch((error) => {
        console.error('Error retrieving files:', error);
      });
  }, []);

  let fileListItem = (
    <Segment
      style={{ marginLeft: '10px', marginRight: '10px', height: '400px' }}
    >
      <Dimmer active inverted>
        <Loader size="large">
          {strings.UPLOAD_FILES_LOADING}
        </Loader>
      </Dimmer>
    </Segment>
  );

  if (fileList) {
    fileListItem = <FilesList fileList={fileList} />;
  }

  return (
    <div>
      <Grid columns={2} relaxed="very" stackable>
        <Grid.Column>
          <Segment textAlign="center" size="huge">
            {strings.UPLOAD_NEW_FILE_TITLE}
          </Segment>
          <NewFile />
        </Grid.Column>
        <Grid.Column>
          <Segment textAlign="center" size="huge">
            {strings.UPLOAD_FILE_TITLE}
          </Segment>
          {fileListItem}
        </Grid.Column>
      </Grid>
      <Divider vertical>OR</Divider>
    </div>
  );
}

export default DataUpload;
