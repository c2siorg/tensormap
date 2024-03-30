import React, { useState, useEffect, useCallback } from "react";
import { Table, Loader } from "semantic-ui-react";
import { getFileData } from "../../services/FileServices";

const DisplayDataset = ({ fileId }) => {
    const [data, setData] = useState(null);

    const fetchData = useCallback(async (fileId) => {
        const fileData = await getFileData(fileId);
        const parsedData = JSON.parse(fileData);
        setData(parsedData);
    }, []);

    useEffect(() => {
        if (fileId) {
            fetchData(fileId);
        }
    }, [fileId, fetchData]);

    return (
        <div>
            {data ? (
                <div style={{ maxHeight: "500px", maxWidth: "100%", overflow: "auto" }}>
                    <Table celled>
                        <Table.Header>
                            <Table.Row>
                                {Object.keys(data[0]).map((key) => (
                                    <Table.HeaderCell key={key}>{key.trim()}</Table.HeaderCell>
                                ))}
                            </Table.Row>
                        </Table.Header>
                        <Table.Body>
                            {data.map((row, index) => (
                                <Table.Row key={index}>
                                    {Object.values(row).map((value, index) => (
                                        <Table.Cell key={index}>{value}</Table.Cell>
                                    ))}
                                </Table.Row>
                            ))}
                        </Table.Body>
                    </Table>
                </div>
            ) : (
                <Loader active>Loading data...</Loader>
            )}
        </div>
    );
};

export default DisplayDataset;