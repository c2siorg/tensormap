import { beforeEach, describe, expect, it, vi } from "vitest";
import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import { downloadCsvFile } from "./FileServices";

vi.mock("../shared/Axios");

describe("downloadCsvFile", () => {
  beforeEach(() => vi.clearAllMocks());

  it("requests the dataset download as a blob and reads its filename", async () => {
    const blob = new Blob(["feature,label\n1,yes\n"], { type: "text/csv" });
    axios.get.mockResolvedValue({
      data: blob,
      headers: { "content-disposition": "attachment; filename*=UTF-8''iris%20data.csv" },
    });

    const result = await downloadCsvFile("file-123");

    expect(axios.get).toHaveBeenCalledWith(`${urls.BACKEND_DOWNLOAD_FILE}file-123/download`, {
      responseType: "blob",
    });
    expect(result).toEqual({ blob, filename: "iris data.csv" });
  });
});
