import React, { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { getFileData } from "../../services/FileServices";
import logger from "../../shared/logger";
import { Button } from "../ui/button";

const DisplayDataset = ({ fileId, pageSize = 50, previewLimit = 100 }) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [pagination, setPagination] = useState(null);
  const abortControllerRef = React.useRef(null);

  const fetchData = useCallback(
    async (currentFileId, currentPage = 1) => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setIsLoading(true);
      try {
        setError(null);
        const fileData = await getFileData(currentFileId, currentPage, pageSize, controller.signal);

        if (!controller.signal.aborted) {
          setData(fileData.data);
          setPagination(fileData.pagination);
        }
      } catch (e) {
        if (e.name !== "CanceledError" && e.name !== "AbortError") {
          logger.error("Error fetching dataset:", e);
          setError("Failed to load dataset");
          setData(null);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      }
    },
    [pageSize],
  );

  useEffect(() => {
    setPage(1);
  }, [fileId]);

  useEffect(() => {
    if (fileId) {
      fetchData(fileId, page);
    }
  }, [fileId, page, fetchData]);

  if (isLoading) {
    return (
      <div className="flex h-48 items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-muted-foreground">Loading data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-48 flex-col items-center justify-center gap-2 text-destructive">
        <span>⚠️ {error}</span>
        <button
          onClick={() => fetchData(fileId, page)}
          className="text-sm text-blue-600 hover:underline"
        >
          Try again
        </button>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        No data available
      </div>
    );
  }

  const displayData = previewLimit ? data.slice(0, previewLimit) : data;
  const hasMore = previewLimit && data.length > previewLimit;

  return (
    <div className="flex flex-col space-y-4">
      {hasMore && (
        <div className="text-sm text-muted-foreground">
          Showing {previewLimit} of {pagination?.total_rows || data.length} rows
        </div>
      )}
      <div className="max-h-[500px] w-full overflow-auto">
        <table className="w-full border-collapse">
          <thead className="sticky top-0 bg-gray-50">
            <tr className="border-b">
              {Object.keys(displayData[0] || {}).map((key) => (
                <th key={key} className="border px-3 py-2 text-left font-semibold bg-gray-50">
                  {key.trim()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayData.map((row, index) => (
              <tr key={index} className="border-b hover:bg-gray-50">
                {Object.values(row).map((value, idx) => (
                  <td key={idx} className="border px-3 py-2">
                    {value === null || value === undefined ? (
                      <span className="text-gray-400">-</span>
                    ) : (
                      String(value)
                    )}
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
  pageSize: PropTypes.number,
  previewLimit: PropTypes.number,
};

DisplayDataset.defaultProps = {
  pageSize: 50,
  previewLimit: 100,
};

export default DisplayDataset;
