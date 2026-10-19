import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

function IssueDetails({ issue }) {
  return (
    <li
      className={`space-y-1 rounded-md border p-3 ${
        issue.blocking
          ? "border-red-300 bg-red-50 text-red-950"
          : "border-amber-300 bg-amber-50 text-amber-950"
      }`}
    >
      <p className="text-xs font-semibold">{issue.blocking ? "Blocking error" : "Warning"}</p>
      <p>{issue.message}</p>
      {issue.columns?.length > 0 && (
        <p className="text-xs font-medium">Affected columns: {issue.columns.join(", ")}</p>
      )}
      {typeof issue.count === "number" && (
        <p className="text-xs">
          Missing values: {issue.count} ({Number(issue.percentage || 0).toFixed(2)}% of rows)
        </p>
      )}
      {typeof issue.distinct_count === "number" && (
        <p className="text-xs">Distinct non-missing values: {issue.distinct_count}</p>
      )}
    </li>
  );
}

export default function DatasetQualityReport({ loading, error, report }) {
  if (loading) {
    return (
      <div
        className="rounded-md border p-4 text-sm text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        Analyzing the selected dataset…
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="border-amber-300 bg-amber-50 text-amber-950">
        <AlertTitle>Dataset quality could not be analyzed</AlertTitle>
        <AlertDescription>
          {error} Existing training validation still applies; you can retry by changing the dataset
          or target.
        </AlertDescription>
      </Alert>
    );
  }

  if (!report) return null;

  if (report.status === "clean") {
    return (
      <div className="space-y-2" aria-live="polite">
        <Alert className="border-emerald-300 bg-emerald-50 text-emerald-950">
          <AlertTitle>Dataset quality looks good</AlertTitle>
          <AlertDescription>
            Checked {report.summary.rows} rows across {report.summary.columns} columns. No quality
            issues were found.
          </AlertDescription>
        </Alert>
        {report.limitations?.map((limitation) => (
          <p className="text-xs text-muted-foreground" key={limitation}>
            {limitation}
          </p>
        ))}
      </div>
    );
  }

  const blocking = report.issues?.some((issue) => issue.blocking);
  const nonInfoIssues = report.issues?.filter((issue) => issue.severity !== "info") || [];
  const infoIssues = report.issues?.filter((issue) => issue.severity === "info") || [];

  return (
    <div className="space-y-3" aria-live="polite">
      {blocking ? (
        <Alert variant="destructive">
          <AlertTitle>Resolve blocking dataset issues before training</AlertTitle>
          <AlertDescription>
            Warnings are informational; the errors below must be addressed to continue.
          </AlertDescription>
        </Alert>
      ) : report.status === "not_applicable" ? (
        <Alert className="border-sky-300 bg-sky-50 text-sky-950">
          <AlertTitle>Tabular quality checks do not apply</AlertTitle>
          <AlertDescription>
            Training will use the existing image dataset validation.
          </AlertDescription>
        </Alert>
      ) : (
        <Alert className="border-amber-300 bg-amber-50 text-amber-950">
          <AlertTitle>Dataset warnings found</AlertTitle>
          <AlertDescription>
            You can continue training, but review the potential data-quality concerns.
          </AlertDescription>
        </Alert>
      )}

      {report.summary && (
        <p className="text-xs text-muted-foreground">
          {report.summary.rows} rows
          {typeof report.summary.usable_rows === "number" &&
            ` (${report.summary.usable_rows} usable)`}
          {`, ${report.summary.columns} columns, ${report.summary.missing_values} missing values`}
        </p>
      )}

      {nonInfoIssues.length > 0 && (
        <ul className="space-y-3 text-sm" aria-label="Dataset quality findings">
          {nonInfoIssues.map((issue, index) => (
            <IssueDetails
              key={`${issue.code}-${issue.columns?.join("-") || index}`}
              issue={issue}
            />
          ))}
        </ul>
      )}
      {infoIssues.map((issue) => (
        <p className="text-sm text-muted-foreground" key={issue.code}>
          {issue.message}
        </p>
      ))}
      {report.limitations?.map((limitation) => (
        <p className="text-xs text-muted-foreground" key={limitation}>
          {limitation}
        </p>
      ))}
    </div>
  );
}
