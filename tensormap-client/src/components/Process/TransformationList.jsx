import { Button, Table } from "semantic-ui-react";

function TransformationList({ transformations, onDelete }) {
    return (
        <Table celled>
            <Table.Header>
                <Table.Row>
                    <Table.HeaderCell width={6}>Feature</Table.HeaderCell>
                    <Table.HeaderCell width={6}>Transformation</Table.HeaderCell>
                    <Table.HeaderCell width={2} textAlign="center">Delete Transformation</Table.HeaderCell>
                </Table.Row>
            </Table.Header>

            <Table.Body>
                {transformations.map((transformation, index) => (
                    <Table.Row key={index}>
                        <Table.Cell>{transformation.feature}</Table.Cell>
                        <Table.Cell>{transformation.transformation}</Table.Cell>
                        <Table.Cell textAlign="center">
                            <Button color="red" size="large" onClick={() => onDelete(index)}>
                                Delete
                            </Button>
                        </Table.Cell>
                    </Table.Row>
                ))}
            </Table.Body>
        </Table>
    );
}

export default TransformationList;