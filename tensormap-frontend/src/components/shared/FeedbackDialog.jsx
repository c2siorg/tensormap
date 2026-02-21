import PropTypes from "prop-types";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { CheckCircle2, AlertCircle } from "lucide-react";
import * as strings from "../../constants/Strings";

export default function FeedbackDialog({ open, onClose, success, message, detail = "" }) {
  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {success ? (
              <CheckCircle2 className="h-5 w-5 text-green-600" />
            ) : (
              <AlertCircle className="h-5 w-5 text-red-600" />
            )}
            {message}
          </DialogTitle>
        </DialogHeader>
        {!success && detail && <p className="text-sm text-muted-foreground">{detail}</p>}
        <DialogFooter>
          <Button variant={success ? "default" : "destructive"} onClick={onClose}>
            {success ? strings.PROCESS_SUCCESS_MODEL_BUTTON : strings.PROCESS_FAIL_MODEL_BUTTON}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

FeedbackDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  success: PropTypes.bool.isRequired,
  message: PropTypes.string.isRequired,
  detail: PropTypes.string,
};
