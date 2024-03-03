import React, { useState, useEffect } from 'react';
import { Table, Loader } from 'semantic-ui-react';
import { getData } from '../../services/FileServices';

const ViewFiles = ({ selectedFile }) => {
  const [data, setData] = useState(null);

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

  return (
    <div>
      <h2>File View</h2>
      {data ? (
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
      ) : (
        <Loader active>Loading data...</Loader>
      )}
    </div>
  );
};

export default ViewFiles;
