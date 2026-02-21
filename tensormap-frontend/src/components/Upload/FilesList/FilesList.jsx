import PropTypes from "prop-types";
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

FilesList.propTypes = {
  fileList: PropTypes.arrayOf(
    PropTypes.shape({
      SavedFileName: PropTypes.string,
      SavedFileType: PropTypes.string,
      SavedFileId: PropTypes.string,
    }),
  ).isRequired,
  setRefresh: PropTypes.func.isRequired,
  refresh: PropTypes.bool.isRequired,
};

export default FilesList;
