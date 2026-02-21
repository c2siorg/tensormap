import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { getFileData } from "../../services/FileServices";
import logger from "../../shared/logger";

/**
 * Tabular preview of a CSV dataset.
 *
 * Fetches file data from the backend and renders it as a scrollable
 * HTML table.
 *
 * @param {{ fileId: string }} props
 */
const DisplayDataset = ({ fileId }) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async (fileId) => {
    try {
      setError(null);
      const fileData = await getFileData(fileId);
      const parsedData = JSON.parse(fileData);
      setData(parsedData);
    } catch (e) {
      logger.error("Error fetching dataset:", e);
      setError("Failed to load dataset");
      setData(null);
    }
  }, []);

  useEffect(() => {
    if (fileId) {
      fetchData(fileId);
    }
  }, [fileId, fetchData]);

  if (error) {
    return <div className="flex h-48 items-center justify-center text-destructive">{error}</div>;
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        Loading data...
      </div>
    );
  }

  return (
    <div className="max-h-[500px] w-full overflow-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b">
            {Object.keys(data[0]).map((key) => (
              <th key={key} className="border px-3 py-2 text-left font-semibold">
                {key.trim()}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, index) => (
            <tr key={index} className="border-b">
              {Object.values(row).map((value, idx) => (
                <td key={idx} className="border px-3 py-2">
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

DisplayDataset.propTypes = {
  fileId: PropTypes.string.isRequired,
};

export default DisplayDataset;
