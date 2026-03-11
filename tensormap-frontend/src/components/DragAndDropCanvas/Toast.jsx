import { useEffect } from "react";
import { X } from "lucide-react";

export default function Toast({ message, onClose }) {
  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => {
        onClose();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [message, onClose]);

  if (!message) return null;

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded shadow-lg animate-in fade-in slide-in-from-top-4">
      <span className="font-medium text-sm">{message}</span>
      <button onClick={onClose} className="p-1 hover:bg-red-200 rounded-full transition-colors">
        <X size={16} />
      </button>
    </div>
  );
}
