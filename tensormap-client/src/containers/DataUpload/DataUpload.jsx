import React, { useEffect, useState } from "react";
import { Grid, Segment, Divider, Dimmer, Loader } from "semantic-ui-react";
import FilesList from "../../components/Upload/FilesList/FilesList";
import * as strings from "../../constants/Strings";
import NewFile from "../../components/Upload/NewFile/NewFile";
import { getAllFiles } from "../../services/FileServices";

function DataUpload() {
    const [fileList, setFileList] = useState(null);
    const [refresh, setRefresh] = useState(false);

    useEffect(() => {
        getAllFiles()
            .then((response) => {
                const files = response.map((item) => ({
                    SavedFileId: item.file_id,
                    SavedFileName: item.file_name,
                    SavedFileType: item.file_type,
                }));
                setFileList(files);
            })
            .catch((error) => {
                console.error("Error retrieving files:", error);
            });
    }, [refresh]);

    let fileListItem = (
        <Segment style={{ marginLeft: "10px", marginRight: "10px", height: "400px" }}>
            <Dimmer active inverted>
                <Loader size="large">
                  {strings.UPLOAD_FILES_LOADING}</Loader>
            </Dimmer>
        </Segment>
    );

    if (fileList) {
        fileListItem = <FilesList fileList={fileList} setRefresh={setRefresh} refresh={refresh}/>;
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
// import React, { Component } from 'react';
// import {
//   Grid, Segment, Divider, Dimmer, Loader,
// } from 'semantic-ui-react';
// import FilesList from '../../components/Upload/FilesList/FilesList';
// import * as strings from '../../constants/Strings';
// import NewFile from '../../components/Upload/NewFile/NewFile';
// import { getAllFiles } from '../../services/FileServices';

// class DataUpload extends Component {
//   state = { fileList: null };

//   componentDidMount() {
//     /*
//         * GET data from backend save them in the states => fileList
//         *
//         * */
//     getAllFiles()
//       .then((response) => {
//         const fileList = response.map((item) => ({
//           SavedFileName: item.file_name,
//           SavedFileType: item.file_type,
//         }));
//         this.setState((prevState) => ({
//           ...prevState,
//           fileList,
//         }));
//       })
//       .catch((error) => {
//         console.error('Error retrieving files:', error);
//       });
//   }

//   render() {
//     /*
//         *  fileItem is a component that can represent as loading component and after file list is loaded to
//         *  state, it will convert to proper file list
//         *
//         * */

//     let fileListItem = (
//       <Segment style={{ marginLeft: '10px', marginRight: '10px', height: '400px' }}>
//         <Dimmer active inverted>
//           <Loader size="large">
//             {strings.UPLOAD_FILES_LOADING}
//             {' '}
//           </Loader>
//         </Dimmer>
//       </Segment>
//     );

//     if (this.state.fileList) {
//       fileListItem = (
//         <FilesList fileList={this.state.fileList} />
//       );
//     }

//     return (
//       <div>
//         <Grid columns={2} relaxed="very" stackable>
//           <Grid.Column>
//             <Segment textAlign="center" size="huge">{strings.UPLOAD_NEW_FILE_TITLE}</Segment>
//             <NewFile />
//           </Grid.Column>
//           <Grid.Column>
//             <Segment textAlign="center" size="huge">{strings.UPLOAD_FILE_TITLE}</Segment>
//             {fileListItem}
//           </Grid.Column>
//         </Grid>
//         <Divider vertical>OR</Divider>
//       </div>
//     );
//   }
// }

// export default DataUpload;
