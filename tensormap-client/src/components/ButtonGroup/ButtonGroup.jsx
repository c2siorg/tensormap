import React from 'react'
import { Button, Confirm } from 'semantic-ui-react';

export const ButtonGroup = () => {

    const handleDelete = (rowIndex) => {
        setShowConfirmDelete(true);
        setModifyRow(rowIndex);
    };

    const handleConfirmDelete = () => {
        // Implement delete action here
        console.log(`Deleting row ${modifyRow}`);
        // Close the confirmation dialog
        setShowConfirmDelete(false);
    };

    const handleCancelDelete = () => {
        // Close the confirmation dialog
        setShowConfirmDelete(false);
    };

    const handleModify = (rowIndex, columnIndex) => {
        setShowModifyInput(true);
        setModifyRow(rowIndex);
        setModifyColumn(columnIndex);
    };

    const handleModifyConfirm = () => {
        // Implement modify action here
        console.log(`Modifying row ${modifyRow} column ${modifyColumn} with value ${modifyValue}`);
        // Clear modify value and hide input
        setModifyValue("");
        setShowModifyInput(false);
    };

    return (
        <>
            <Button.Group>
                <Button icon="trash" onClick={() => handleDelete(modifyRow)}>Delete</Button>
                <Button icon="edit" onClick={() => handleModify(modifyRow, modifyColumn)}>Modify</Button>
            </Button.Group>
            <Confirm
                open={showConfirmDelete}
                onCancel={handleCancelDelete}
                onConfirm={handleConfirmDelete}
                content={`Are you sure you want to delete row ${modifyRow + 1}?`}
            />
        </>
    )
}
