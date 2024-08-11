import { useState } from "react";
import { Button, Table, Input, Select } from "semantic-ui-react";
import { submitImageParameters } from "../../services/FileServices";

const colorModeOptions = [
    { key: "grayscale", value: "grayscale", text: "Grayscale" },
    { key: "rgb", value: "rgb", text: "RGB" },
    { key: "rgba", value: "rgba", text: "RGBA" },
];

const labelModeOptions = [
    { key: "int", value: "int", text: "Int" },
    { key: "categorical", value: "categorical", text: "Categorical" },
    { key: "binary", value: "binary", text: "Binary" },
];

const ImagePreprocess = ({ fileId, fileType }) => {
    const [imageSize, setImageSize] = useState("");
    const [batchSize, setBatchSize] = useState("");
    const [colorMode, setColorMode] = useState("rgb");
    const [labelMode, setLabelMode] = useState("int");

    const handleSubmit = (e) => {
        e.preventDefault();
        const data = {
            fileId: fileId,
            fileType: fileType, 
            image_size: parseInt(imageSize, 10),
            batch_size: parseInt(batchSize, 10),
            color_mode: colorMode,
            label_mode: labelMode,
        };
        submitImageParameters(data)
    };

    return (
        <form onSubmit={handleSubmit}>
            <Table celled>
                <Table.Header>
                    <Table.Row>
                        <Table.HeaderCell>Image Size</Table.HeaderCell>
                        <Table.HeaderCell>Batch Size</Table.HeaderCell>
                        <Table.HeaderCell>Color Mode</Table.HeaderCell>
                        <Table.HeaderCell>Label Mode</Table.HeaderCell>
                        <Table.HeaderCell>Submit</Table.HeaderCell>
                    </Table.Row>
                </Table.Header>

                <Table.Body>
                    <Table.Row>
                        <Table.Cell>
                            <Input
                                type="number"
                                value={imageSize}
                                onChange={(e) => setImageSize(e.target.value)}
                                required
                            />
                        </Table.Cell>
                        <Table.Cell>
                            <Input
                                type="number"
                                value={batchSize}
                                onChange={(e) => setBatchSize(e.target.value)}
                                required
                            />
                        </Table.Cell>
                        <Table.Cell>
                            <Select
                                options={colorModeOptions}
                                value={colorMode}
                                onChange={(e, { value }) => setColorMode(value)}
                                required
                            />
                        </Table.Cell>
                        <Table.Cell>
                            <Select
                                options={labelModeOptions}
                                value={labelMode}
                                onChange={(e, { value }) => setLabelMode(value)}
                                required
                            />
                        </Table.Cell>
                        <Table.Cell textAlign="center">
                            <Button type="submit" color="green" size="large">
                                Submit
                            </Button>
                        </Table.Cell>
                    </Table.Row>
                </Table.Body>
            </Table>
        </form>
    );
};

export default ImagePreprocess;
