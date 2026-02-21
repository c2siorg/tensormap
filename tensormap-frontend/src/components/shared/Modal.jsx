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

function ModalComponent({ modalOpen, modelClose, success, Modalmessage, modalText = "" }) {
  const successModelButton = strings.PROCESS_SUCCESS_MODEL_BUTTON;
  const failModelButton = strings.PROCESS_FAIL_MODEL_BUTTON;

  return (
    <Dialog open={modalOpen} onOpenChange={(isOpen) => !isOpen && modelClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {success ? (
              <CheckCircle2 className="h-5 w-5 text-green-600" />
            ) : (
              <AlertCircle className="h-5 w-5 text-red-600" />
            )}
            {Modalmessage}
          </DialogTitle>
        </DialogHeader>
        {!success && modalText && <p className="text-sm text-muted-foreground">{modalText}</p>}
        <DialogFooter>
          <Button variant={success ? "default" : "destructive"} onClick={modelClose}>
            {success ? successModelButton : failModelButton}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

ModalComponent.propTypes = {
  modalOpen: PropTypes.bool.isRequired,
  modelClose: PropTypes.func.isRequired,
  success: PropTypes.bool.isRequired,
  Modalmessage: PropTypes.string.isRequired,
  modalText: PropTypes.string,
};

export default ModalComponent;
