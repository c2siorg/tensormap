import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { Button } from "@/components/ui/button";
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
  const PAGE_SIZE = 50;
  const [rows, setRows] = useState([]);
  const [columns, setColumns] = useState([]);
  const [pagination, setPagination] = useState({ total: 0, offset: 0, limit: PAGE_SIZE });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(
    async (nextOffset = 0) => {
      try {
        setLoading(true);
        setError(null);
        const fileData = await getFileData(fileId, { offset: nextOffset, limit: PAGE_SIZE });
        setRows(fileData.rows || []);
        setColumns(fileData.columns || []);
        setPagination(fileData.pagination || { total: 0, offset: nextOffset, limit: PAGE_SIZE });
      } catch (e) {
        logger.error("Error fetching dataset:", e);
        setError("Failed to load dataset");
        setRows([]);
        setColumns([]);
      } finally {
        setLoading(false);
      }
    },
    [fileId],
  );

  useEffect(() => {
    if (fileId) {
      fetchData(0);
    }
  }, [fileId, fetchData]);

  if (error) {
    return <div className="flex h-48 items-center justify-center text-destructive">{error}</div>;
  }

  if (loading) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        Loading data...
      </div>
    );
  }

  if (!rows.length) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        No rows found.
      </div>
    );
  }

  const { total, offset, limit } = pagination;
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + rows.length, total);
  const hasPrev = offset > 0;
  const hasNext = offset + limit < total;

  const handlePrevious = () => {
    if (!hasPrev) return;
    fetchData(Math.max(0, offset - limit));
  };

  const handleNext = () => {
    if (!hasNext) return;
    fetchData(offset + limit);
  };

  return (
    <div className="space-y-3">
      <div className="max-h-[500px] w-full overflow-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b">
              {columns.map((key) => (
                <th key={key} className="border px-3 py-2 text-left font-semibold">
                  {key.trim()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${offset}-${index}`} className="border-b">
                {columns.map((column) => (
                  <td key={column} className="border px-3 py-2">
                    {String(row[column] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          Showing {start}-{end} of {total}
        </span>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePrevious}
            disabled={!hasPrev || loading}
          >
            Previous
          </Button>
          <Button variant="outline" size="sm" onClick={handleNext} disabled={!hasNext || loading}>
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
