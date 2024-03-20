import React, { useState, useEffect } from "react";
import { Table, Loader, Button, Icon, Input, Confirm } from "semantic-ui-react";
import { getData } from "../../services/FileServices";

const ViewFiles = ({ selectedFile }) => {
  const [data, setData] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [modifyValue, setModifyValue] = useState("");
  const [modifyRow, setModifyRow] = useState(null);
  const [modifyColumn, setModifyColumn] = useState(null);
  const [showModifyInput, setShowModifyInput] = useState(false);
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);
  const [enableModify, setEnableModify] = useState(false);
  const [deleteInput, setDeleteInput] = useState(""); // Input for delete row
  const [modifyRowInput, setModifyRowInput] = useState(""); // Input for modify row
  const [modifyColumnInput, setModifyColumnInput] = useState(""); // Input for modify column
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false); // Delete confirmation dialog
  const [showModifyConfirmation, setShowModifyConfirmation] = useState(false); // Modify confirmation dialog
  const itemsPerPage = 25;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const fileData = await getData(selectedFile);
        // Parse the JSON data
        const parsedData = JSON.parse(fileData);
        setData(parsedData);
      } catch (error) {
        console.error(error);
      }
    };
    fetchData();
  }, [selectedFile]);

  // Calculate total number of pages
  const totalPages = data ? Math.ceil(data.length / itemsPerPage) : 0;

  // Calculate index range for current page
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = currentPage * itemsPerPage;

  // Slice data for current page
  const currentPageData = data ? data.slice(startIndex, endIndex) : [];

  const handleNextPage = () => {
    setCurrentPage(currentPage + 1);
  };

  const handlePrevPage = () => {
    setCurrentPage(currentPage - 1);
  };

  const handleDelete = () => {
    if (!deleteInput || isNaN(deleteInput) || deleteInput < 1 || deleteInput > data.length) {
      alert("Please enter a valid row index to delete.");
      return;
    }
    setShowDeleteConfirmation(true);
  };

  const handleConfirmDelete = () => {
    // Implement delete action here
    console.log(`Deleting row ${deleteInput}`);
    // Close the confirmation dialog
    setShowConfirmDelete(false);
    setShowDeleteConfirmation(false);
  };

  const handleCancelDelete = () => {
    // Close the confirmation dialog
    setShowDeleteConfirmation(false);
  };

  const handleModify = () => {
    if (!modifyRowInput || isNaN(modifyRowInput) || modifyRowInput < 1 || modifyRowInput > data.length) {
      alert("Please enter a valid row index to modify.");
      return;
    }
    if (!modifyColumnInput || isNaN(modifyColumnInput) || modifyColumnInput < 1 || modifyColumnInput > Object.keys(data[0]).length) {
      alert("Please enter a valid column index to modify.");
      return;
    }
    setShowModifyConfirmation(true);
  };

  const handleConfirmModify = () => {
    // Implement modify action here
    console.log(`Modifying row ${modifyRowInput} column ${modifyColumnInput} with value ${modifyValue}`);
    // Clear modify value and hide input
    setModifyValue("");
    setShowModifyInput(false);
    setShowModifyConfirmation(false);
  };

  const handleCancelModify = () => {
    // Close the confirmation dialog
    setShowModifyConfirmation(false);
  };

  const handleToggleModify = () => {
    setEnableModify(!enableModify);
  };

  return (
    <div>
      <h2>File View</h2>
      <Button onClick={handleToggleModify}>{enableModify ? 'Disable Modify' : 'Enable Modify'}</Button>
      {enableModify && (
        <>
          <Input
            placeholder="Row to Delete"
            value={deleteInput}
            onChange={(e) => setDeleteInput(e.target.value)}
            style={{ marginRight: "10px" }}
          />
          <Button onClick={handleDelete}>Delete</Button>
          <Input
            placeholder="Row to Modify"
            value={modifyRowInput}
            onChange={(e) => setModifyRowInput(e.target.value)}
            style={{ marginRight: "10px", marginLeft: "10px" }}
          />
          <Input
            placeholder="Column to Modify"
            value={modifyColumnInput}
            onChange={(e) => setModifyColumnInput(e.target.value)}
            style={{ marginRight: "10px" }}
          />
          <Button onClick={handleModify}>Modify</Button>
        </>
      )}
      {data ? (
        <>
          <Table celled>
            <Table.Header>
              <Table.Row>
                <Table.HeaderCell>S.no</Table.HeaderCell> {/* Row number column */}
                {Object.keys(data[0]).map((key) => (
                  <Table.HeaderCell key={key}>{key.trim()}</Table.HeaderCell>
                ))}
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {currentPageData.map((row, rowIndex) => (
                <Table.Row key={rowIndex}>
                  <Table.Cell>{startIndex + rowIndex + 1}</Table.Cell> {/* Row number */}
                  {Object.entries(row).map(([key, value], columnIndex) => (
                    <Table.Cell key={columnIndex}>
                      {/* Clickable Cell */}
                      {showModifyInput && modifyRow === startIndex + rowIndex && modifyColumn === columnIndex ? (
                        <Input
                          value={modifyValue}
                          onChange={(e) => setModifyValue(e.target.value)}
                          action={{
                            icon: "check",
                            onClick: handleModifyConfirm,
                          }}
                        />
                      ) : (
                        <span onClick={() => enableModify && handleModify(rowIndex, columnIndex)}>{value}</span>
                      )}
                    </Table.Cell>
                  ))}
                </Table.Row>
              ))}
            </Table.Body>
          </Table>
          <div style={{ textAlign: "center" }}>
            <Button icon labelPosition="left" onClick={handlePrevPage} disabled={currentPage === 1}>
              <Icon name="arrow left" />
              Previous
            </Button>
            <span style={{ margin: "0 10px" }}>
              Page {currentPage} of {totalPages}
            </span>
            <Button icon labelPosition="right" onClick={handleNextPage} disabled={currentPage === totalPages}>
              Next
              <Icon name="arrow right" />
            </Button>
          </div>
          <Confirm
            open={showConfirmDelete}
            onCancel={handleCancelDelete}
            onConfirm={handleConfirmDelete}
            content={`Are you sure you want to delete row ${deleteInput}?`}
          />
          <Confirm
            open={showModifyConfirmation}
            onCancel={handleCancelModify}
            onConfirm={handleConfirmModify}
            content={`Are you sure you want to modify row ${modifyRowInput} column ${modifyColumnInput}?`}
          />
        </>
      ) : (
        <Loader active>Loading data...</Loader>
      )}
    </div>
  );
};

export default ViewFiles;
