import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { getColumnStats } from "../../services/FileServices";
import logger from "../../shared/logger";

/**
 * Collapsible panel showing per-column descriptive statistics.
 *
 * Collapsed by default. Shows total row/column count in the header.
 * Numeric columns show mean, min, max; non-numeric columns show "—".
 *
 * @param {{ fileId: string }} props
 */
const fmt = (val) => {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") return Number.isInteger(val) ? val : val.toFixed(4);
  return String(val);
};

const ColumnStatsPanel = ({ fileId }) => {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(false);

  const fetchStats = useCallback(async (id) => {
    try {
      setError(null);
      const data = await getColumnStats(id);
      setStats(data);
    } catch (e) {
      logger.error("Error fetching column stats:", e);
      setError("Failed to load column statistics");
      setStats(null);
    }
  }, []);

  useEffect(() => {
    if (fileId) {
      setStats(null);
      fetchStats(fileId);
    } else {
      setStats(null);
    }
  }, [fileId, fetchStats]);

  const summary =
    stats?.total_rows != null && stats?.total_cols != null
      ? `${stats.total_rows} rows · ${stats.total_cols} columns`
      : "";

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between py-2 text-sm font-medium text-muted-foreground hover:text-foreground"
      >
        <span>
          Column Statistics
          {summary && (
            <span className="ml-2 font-normal text-xs text-muted-foreground">
              ({summary})
            </span>
          )}
        </span>
        <span>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <>
          {error && (
            <div className="flex h-16 items-center justify-center text-destructive text-sm">
              {error}
            </div>
          )}
          {!error && !stats && (
            <div className="flex h-16 items-center justify-center text-muted-foreground text-sm">
              Loading statistics...
            </div>
          )}
          {!error && stats && Array.isArray(stats.columns) && stats.columns.length === 0 && (
            <div className="flex h-16 items-center justify-center text-muted-foreground text-sm">
              No statistics available
            </div>
          )}
          {!error && stats && Array.isArray(stats.columns) && stats.columns.length > 0 && (
            <div className="mt-2 overflow-auto max-h-[400px]">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    {["Column", "Type", "Non-Null", "Null", "Mean", "Min", "Max"].map((h) => (
                      <th key={h} className="border px-3 py-2 text-left font-semibold">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {stats.columns.map((row) => (
                    <tr key={row.column} className="border-b hover:bg-muted/30">
                      <td className="border px-3 py-2 font-medium">{row.column}</td>
                      <td className="border px-3 py-2 text-muted-foreground">{row.dtype}</td>
                      <td className="border px-3 py-2 text-muted-foreground">{fmt(row.count)}</td>
                      <td className="border px-3 py-2 text-muted-foreground">{fmt(row.null_count)}</td>
                      <td className="border px-3 py-2 text-muted-foreground">{fmt(row.mean)}</td>
                      <td className="border px-3 py-2 text-muted-foreground">{fmt(row.min)}</td>
                      <td className="border px-3 py-2 text-muted-foreground">{fmt(row.max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
};

ColumnStatsPanel.propTypes = {
  fileId: PropTypes.string.isRequired,
};

export default ColumnStatsPanel;
