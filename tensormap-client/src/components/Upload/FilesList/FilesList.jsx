import File from "./File/File";

function FilesList(props) {
  const fileList = props.fileList.map((file, index) => (
    <File
      key={index}
      savedFileName={file.SavedFileName}
      savedFileType={file.SavedFileType}
      savedFileId={file.SavedFileId}
      setRefresh={props.setRefresh}
      refresh={props.refresh}
    />
  ));
  return <div>{fileList}</div>;
}

export default FilesList;
