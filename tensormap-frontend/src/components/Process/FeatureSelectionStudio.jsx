import { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { Button } from "@/components/ui/button";
import { analyzeFeatures, applyFeatureSelection } from "../../services/FeatureSelectionService";

const formatScore = (score) => Number(score).toFixed(4);

function FeatureSelectionStudio({ fileId, datasetRevision, columns, updateFileList }) {
  const [target, setTarget] = useState("");
  const [task, setTask] = useState("auto");
  const [profile, setProfile] = useState(null);
  const [columnsToDrop, setColumnsToDrop] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    setTarget("");
    setProfile(null);
    setColumnsToDrop([]);
    setError("");
    setMessage("");
  }, [fileId]);

  useEffect(() => {
    setProfile(null);
    setColumnsToDrop([]);
  }, [datasetRevision]);

  useEffect(() => {
    if (target && !columns.includes(target)) {
      setTarget("");
      setProfile(null);
      setColumnsToDrop([]);
    }
  }, [columns, target]);

  const handleAnalyze = async () => {
    if (!target) {
      setError("Choose the target column before analyzing features.");
      return;
    }
    setLoading(true);
    setError("");
    setMessage("");
    setColumnsToDrop([]);
    try {
      setProfile(await analyzeFeatures(fileId, target, task));
    } catch (e) {
      setProfile(null);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleApply = async () => {
    if (!columnsToDrop.length) return;
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const response = await applyFeatureSelection(fileId, target, columnsToDrop);
      setMessage(
        `${response.data.removed_columns.length} feature(s) removed. Backup: ${response.data.backup_file}`,
      );
      setProfile(null);
      setColumnsToDrop([]);
      updateFileList();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleColumn = (feature) => {
    setColumnsToDrop((selected) =>
      selected.includes(feature)
        ? selected.filter((column) => column !== feature)
        : [...selected, feature],
    );
  };

  return (
    <section className="space-y-4" aria-label="Feature selection studio">
      <p className="text-sm text-muted-foreground">
        Rank predictors with mutual information, ANOVA F-statistics, and variance. The composite
        score is the mean of each metric normalized across the dataset.
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <label className="grid gap-1 text-sm font-medium">
          Target column
          <select
            aria-label="Target column"
            value={target}
            disabled={loading}
            onChange={(event) => {
              setTarget(event.target.value);
              setProfile(null);
              setColumnsToDrop([]);
            }}
            className="h-10 min-w-48 rounded-md border bg-background px-3"
          >
            <option value="">Select target</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-sm font-medium">
          Target type
          <select
            aria-label="Target type"
            value={task}
            disabled={loading}
            onChange={(event) => {
              setTask(event.target.value);
              setProfile(null);
              setColumnsToDrop([]);
            }}
            className="h-10 rounded-md border bg-background px-3"
          >
            <option value="auto">Auto-detect</option>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </label>
        <Button type="button" onClick={handleAnalyze} disabled={loading || columns.length < 2}>
          {loading ? "Working..." : "Analyze relevance"}
        </Button>
      </div>

      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {message && (
        <p role="status" className="text-sm text-green-700">
          {message}
        </p>
      )}
      {profile && (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            {profile.task === "classification" ? "Classification" : "Regression"} ·{" "}
            {profile.sample_rows} target rows · Target: {profile.target}
          </p>
          {profile.scores.length === 0 ? (
            <p className="text-sm text-muted-foreground">No predictor columns are available.</p>
          ) : (
            <div className="max-h-[420px] overflow-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th scope="col" className="px-3 py-2 text-left">
                      Rank
                    </th>
                    <th scope="col" className="px-3 py-2 text-left">
                      Feature
                    </th>
                    <th scope="col" className="px-3 py-2 text-right">
                      MI
                    </th>
                    <th scope="col" className="px-3 py-2 text-right">
                      ANOVA F
                    </th>
                    <th scope="col" className="px-3 py-2 text-right">
                      Variance
                    </th>
                    <th scope="col" className="px-3 py-2 text-right">
                      Composite
                    </th>
                    <th scope="col" className="px-3 py-2 text-left">
                      Remove
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {profile.scores.map((score, index) => (
                    <tr key={score.feature} className="border-b">
                      <td className="px-3 py-2">{index + 1}</td>
                      <td className="px-3 py-2 font-medium">{score.feature}</td>
                      <td className="px-3 py-2 text-right">
                        {formatScore(score.mutual_information)}
                      </td>
                      <td className="px-3 py-2 text-right">{formatScore(score.anova_f)}</td>
                      <td className="px-3 py-2 text-right">{formatScore(score.variance)}</td>
                      <td className="px-3 py-2 text-right">{formatScore(score.composite_score)}</td>
                      <td className="px-3 py-2">
                        <input
                          type="checkbox"
                          aria-label={`Remove ${score.feature}`}
                          checked={columnsToDrop.includes(score.feature)}
                          onChange={() => toggleColumn(score.feature)}
                          disabled={loading}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <Button
            type="button"
            variant="destructive"
            onClick={handleApply}
            disabled={loading || columnsToDrop.length === 0}
          >
            Remove selected features
          </Button>
          <p className="text-xs text-muted-foreground">
            Applying creates or replaces a sibling .bak file before updating the dataset.
          </p>
        </div>
      )}
    </section>
  );
}

FeatureSelectionStudio.propTypes = {
  fileId: PropTypes.string.isRequired,
  datasetRevision: PropTypes.number.isRequired,
  columns: PropTypes.arrayOf(PropTypes.string).isRequired,
  updateFileList: PropTypes.func.isRequired,
};

export default FeatureSelectionStudio;
