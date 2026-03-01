import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { getFileData } from "../../services/FileServices";
import logger from "../../shared/logger";
import { Button } from "../ui/button";

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
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [pagination, setPagination] = useState(null);

  const fetchData = useCallback(async (fileId, currentPage = 1) => {
    setIsLoading(true);
    try {
      setError(null);
      const fileData = await getFileData(fileId, currentPage, 50);

      let parsedData;
      if (typeof fileData.data === "string") {
        parsedData = JSON.parse(fileData.data);
      } else {
        parsedData = fileData.data;
      }

      setData(parsedData);
      setPagination(fileData.pagination);
    } catch (e) {
      logger.error("Error fetching dataset:", e);
      setError("Failed to load dataset");
      setData(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (fileId) {
      fetchData(fileId, page);
    }
  }, [fileId, page, fetchData]);

  if (error) {
    return <div className="flex h-48 items-center justify-center text-destructive">{error}</div>;
  }

  if (isLoading) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        Loading data...
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        Dataset is empty.
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-4">
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
                    {value !== null ? String(value) : ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <div>
          Showing {pagination ? (pagination.page - 1) * pagination.page_size + 1 : 0} to{" "}
          {pagination ? Math.min(pagination.page * pagination.page_size, pagination.total_rows) : 0}{" "}
          of {pagination ? pagination.total_rows : 0} Total rows
        </div>
        <div className="flex items-center space-x-2">
          <div>
            Page {pagination?.page || 1} of {pagination?.total_pages || 1}
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={!pagination || pagination.page <= 1}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => p + 1)}
            disabled={!pagination || pagination.page >= pagination.total_pages}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
};

DisplayDataset.propTypes = {
  fileId: PropTypes.string.isRequired,
};

export default DisplayDataset;
