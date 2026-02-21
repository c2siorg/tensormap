import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { getFileData } from "../../services/FileServices";

const DisplayDataset = ({ fileId, fileType }) => {
  const [data, setData] = useState(null);

  const fetchData = useCallback(async (fileId) => {
    const fileData = await getFileData(fileId);
    const parsedData = JSON.parse(fileData);
    setData(parsedData);
  }, []);

  useEffect(() => {
    if (fileId && fileType !== "zip") {
      fetchData(fileId);
    }
  }, [fileId, fileType, fetchData]);

  return fileType === "zip" ? (
    <div className="p-4 text-muted-foreground">ZIP files currently cannot be displayed</div>
  ) : (
    <div>
      {data && data.length > 0 ? (
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
      ) : (
        <div className="flex h-48 items-center justify-center text-muted-foreground">
          Loading data...
        </div>
      )}
    </div>
  );
};

DisplayDataset.propTypes = {
  fileId: PropTypes.string.isRequired,
  fileType: PropTypes.string.isRequired,
};

export default DisplayDataset;
