import { FileText } from "lucide-react";

function SelectFileModal() {
  return (
    <div className="flex h-[50vh] flex-col items-center justify-center text-center">
      <FileText className="mb-4 h-12 w-12 text-muted-foreground" />
      <h2 className="text-xl font-semibold">Select a File</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Select a dataset to view the metrics of the Dataset
      </p>
    </div>
  );
}

export default SelectFileModal;
