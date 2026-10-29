import { useState } from "react";
import PropTypes from "prop-types";
import { Download } from "lucide-react";
import { Button } from "../ui/button";
import { downloadCsvFile } from "../../services/FileServices";
import logger from "../../shared/logger";

const fallbackCsvName = (name) => {
  const baseName = (name || "dataset").replace(/\.csv$/i, "");
  const safeName = baseName.replace(/[\\/:*?"<>|]/g, "_").trim() || "dataset";
  return `${safeName}.csv`;
};

function DownloadCsvButton({ fileId, fileName, fileType }) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");

  if (!fileId || fileType?.toLowerCase() !== "csv") return null;

  const handleDownload = async () => {
    setDownloading(true);
    setError("");
    let objectUrl;
    try {
      const result = await downloadCsvFile(fileId);
      objectUrl = window.URL.createObjectURL(result.blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = result.filename || fallbackCsvName(fileName);
      anchor.style.display = "none";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    } catch (downloadError) {
      logger.error("CSV download failed:", downloadError);
      setError("Could not download this CSV. Please try again.");
    } finally {
      if (objectUrl) window.setTimeout(() => window.URL.revokeObjectURL(objectUrl), 0);
      setDownloading(false);
    }
  };

  return (
    <div className="flex flex-col items-start gap-1">
      <Button type="button" variant="outline" onClick={handleDownload} disabled={downloading}>
        <Download className="mr-2 h-4 w-4" />
        {downloading ? "Downloading..." : "Download CSV"}
      </Button>
      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

DownloadCsvButton.propTypes = {
  fileId: PropTypes.string,
  fileName: PropTypes.string,
  fileType: PropTypes.string,
};

export default DownloadCsvButton;
