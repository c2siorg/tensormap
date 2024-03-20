import React from "react";
import { Segment, Button } from "semantic-ui-react";
import { deleteFile } from "../../../../services/FileServices";
import { useState } from "react";
function File(props) {

    const deleteFileHandeler = async () => {
      const a = await deleteFile(props.savedFileId)
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
