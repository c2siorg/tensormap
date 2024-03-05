import React from 'react';
import File from './File/File';

function FilesList(props) {
  const fileList = props.fileList.map((file, index) => (
    <File
      key={index}
      savedFileName={file.SavedFileName}
      savedFileType={file.SavedFileType}
    />
  ));
  return (
    <div>
      {fileList}
    </div>
  );
}

export default FilesList;
