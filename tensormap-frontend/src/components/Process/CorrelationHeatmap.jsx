import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { getCorrelationMatrix } from "../../services/FileServices";
import logger from "../../shared/logger";

/**
 * Converts a correlation value in [-1, 1] to an HSL background colour.
 *
 * positive → red (#ee4444), zero → white (#ffffff), negative → blue (#4444ee)
 *
 * @param {number|null} value
 * @returns {string} CSS colour string
 */
function correlationColour(value) {
  if (value === null || value === undefined) return "#e5e7eb"; // muted grey for NaN
  const v = Math.max(-1, Math.min(1, value));
  if (v >= 0) {
    // white → red
    const r = 255;
    const g = Math.round(255 * (1 - v));
    const b = Math.round(255 * (1 - v));
    return `rgb(${r},${g},${b})`;
  } else {
    // white → blue
    const abs = -v;
    const r = Math.round(255 * (1 - abs));
    const g = Math.round(255 * (1 - abs));
    const b = 255;
    return `rgb(${r},${g},${b})`;
  }
}

/**
 * Fetches and displays the pairwise correlation matrix for a dataset as a
 * colour-coded heatmap.
 *
 * - Positive correlations are shown in red, negative in blue, zero in white.
 * - Hovering a cell shows a tooltip with the exact value and column pair.
 * - The grid is scrollable when there are many columns.
 *
 * @param {{ fileId: string }} props
 */
const CorrelationHeatmap = ({ fileId }) => {
  const [data, setData] = useState(null);   // { columns, matrix }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tooltip, setTooltip] = useState(null); // { x, y, row, col, value }
  const [visible, setVisible] = useState(false);

  const fetchMatrix = useCallback(async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const result = await getCorrelationMatrix(fileId);
      setData(result);
      setVisible(true);
    } catch (e) {
      logger.error("Error fetching correlation matrix:", e);
      setError("Failed to load correlation matrix.");
    } finally {
      setLoading(false);
    }
  }, [fileId]);

  // Reset when file changes
  useEffect(() => {
    setVisible(false);
    setData(null);
    setError(null);
  }, [fileId]);

  const CELL = 40; // px — cell size

  return (
    <div className="space-y-3">
      {!visible && (
        <button
          onClick={fetchMatrix}
          disabled={loading}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
        >
          {loading ? "Computing…" : "Show Correlation Heatmap"}
        </button>
      )}

      {error && (
        <p className="text-sm text-destructive">{error}</p>
      )}

      {visible && data && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-muted-foreground">
              {data.columns.length > 0
                ? `${data.columns.length} numeric column${data.columns.length !== 1 ? "s" : ""}`
                : "No numeric columns"}
            </p>
            <button
              onClick={() => setVisible(false)}
              className="text-xs text-muted-foreground underline hover:text-foreground"
            >
              Hide
            </button>
          </div>

          {data.columns.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              This dataset has no numeric columns to correlate.
            </p>
          ) : (
            <div className="relative overflow-auto rounded border">
              {/* Tooltip */}
              {tooltip && (
                <div
                  className="pointer-events-none fixed z-50 rounded bg-gray-900 px-2 py-1 text-xs text-white shadow"
                  style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
                >
                  <span className="font-semibold">{tooltip.col}</span>
                  {" × "}
                  <span className="font-semibold">{tooltip.row}</span>
                  <br />
                  {tooltip.value !== null ? tooltip.value.toFixed(4) : "NaN"}
                </div>
              )}

              <table className="border-collapse text-xs" style={{ minWidth: (data.columns.length + 1) * CELL }}>
                <thead>
                  <tr>
                    {/* top-left empty corner */}
                    <th style={{ width: CELL, height: CELL }} />
                    {data.columns.map((col) => (
                      <th
                        key={col}
                        title={col}
                        style={{ width: CELL, height: CELL, maxWidth: CELL }}
                        className="overflow-hidden border p-0 text-center font-medium"
                      >
                        <div
                          className="overflow-hidden text-ellipsis whitespace-nowrap px-1"
                          style={{ maxWidth: CELL }}
                        >
                          {col}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.matrix.map((row, ri) => (
                    <tr key={ri}>
                      {/* row label */}
                      <td
                        title={data.columns[ri]}
                        style={{ width: CELL, height: CELL, maxWidth: CELL }}
                        className="overflow-hidden border p-0 font-medium"
                      >
                        <div
                          className="overflow-hidden text-ellipsis whitespace-nowrap px-1"
                          style={{ maxWidth: CELL }}
                        >
                          {data.columns[ri]}
                        </div>
                      </td>
                      {row.map((val, ci) => (
                        <td
                          key={ci}
                          style={{
                            width: CELL,
                            height: CELL,
                            backgroundColor: correlationColour(val),
                            cursor: "default",
                          }}
                          className="border p-0 text-center"
                          onMouseMove={(e) =>
                            setTooltip({
                              x: e.clientX,
                              y: e.clientY,
                              row: data.columns[ri],
                              col: data.columns[ci],
                              value: val,
                            })
                          }
                          onMouseLeave={() => setTooltip(null)}
                        />
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Legend */}
              <div className="flex items-center gap-3 border-t px-3 py-2 text-xs text-muted-foreground">
                <span
                  className="inline-block h-3 w-3 rounded-sm"
                  style={{ background: "rgb(68,68,238)" }}
                />
                −1 (negative)
                <span
                  className="inline-block h-3 w-3 rounded-sm border"
                  style={{ background: "#ffffff" }}
                />
                0
                <span
                  className="inline-block h-3 w-3 rounded-sm"
                  style={{ background: "rgb(238,68,68)" }}
                />
                +1 (positive)
                <span
                  className="inline-block h-3 w-3 rounded-sm"
                  style={{ background: "#e5e7eb" }}
                />
                NaN
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

CorrelationHeatmap.propTypes = {
  fileId: PropTypes.string.isRequired,
};

export default CorrelationHeatmap;
