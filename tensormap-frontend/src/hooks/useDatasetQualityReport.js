import { useEffect, useRef, useState } from "react";
import { getDatasetQualityReport } from "../services/FileServices";

/** Load a dataset report for the current file, target, and task without stale results. */
export default function useDatasetQualityReport(fileId, targetField, problemTypeId, trainingSplit) {
  const [state, setState] = useState(null);
  const requestRef = useRef(0);
  const key = JSON.stringify([
    fileId || "",
    targetField || "",
    problemTypeId || "",
    trainingSplit || "",
  ]);
  const currentState = state?.key === key ? state : null;

  useEffect(() => {
    const requestId = ++requestRef.current;
    if (!fileId) {
      setState(null);
      return undefined;
    }

    const controller = new AbortController();
    setState({ key, loading: true, report: null, error: null });
    const timeoutId = setTimeout(() => {
      getDatasetQualityReport(fileId, targetField, problemTypeId, trainingSplit, controller.signal)
        .then((report) => {
          if (requestId === requestRef.current) {
            setState({ key, loading: false, report, error: null });
          }
        })
        .catch((error) => {
          if (requestId !== requestRef.current || controller.signal.aborted) return;
          setState({
            key,
            loading: false,
            report: null,
            error: error?.response?.data?.message || error?.message || "Please try again.",
          });
        });
    }, 300);

    return () => {
      clearTimeout(timeoutId);
      controller.abort();
      if (requestRef.current === requestId) requestRef.current += 1;
    };
  }, [fileId, targetField, problemTypeId, trainingSplit, key]);

  return {
    loading: Boolean(fileId) && (!currentState || currentState.loading),
    error: currentState?.error || null,
    report: currentState?.report || null,
  };
}
