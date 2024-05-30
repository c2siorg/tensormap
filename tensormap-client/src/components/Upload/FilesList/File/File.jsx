import { Segment, Button } from "semantic-ui-react";
import { deleteFile } from "../../../../services/FileServices";
function File(props) {

    const deleteFileHandeler = async () => {
      await deleteFile(props.savedFileId)
      props.setRefresh(!props.refresh)
    } 
    return (
        <div>
            <Segment.Group horizontal raised>
                <Segment>{props.savedFileName}</Segment>
                <Segment>{props.savedFileType}</Segment>
                <Segment style={{ textAlign: "right" }}>
                    <Button negative onClick={deleteFileHandeler}>Delete Dataset</Button>
                </Segment>
            </Segment.Group>
        </div>
    );
}

export default File;
