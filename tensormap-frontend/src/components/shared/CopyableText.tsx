/**
 * CopyableText component - displays text with a copy-to-clipboard button.
 * @module
 */

import { useState } from "react";
import { Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import logger from "@/shared/logger";

interface CopyableTextProps {
  value: string;
  label?: string;
  truncate?: boolean;
  className?: string;
}

export default function CopyableText({
  value,
  label,
  truncate = true,
  className = "",
}: CopyableTextProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      logger.error("Failed to copy to clipboard:", err);
      // Do not show success feedback on failure
    }
  };

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span
        className={`font-mono text-sm ${truncate ? "truncate" : "break-all"}`}
        title={value}
      >
        {value}
      </span>
      <Button
        variant="ghost"
        size="sm"
        onClick={handleCopy}
        className="h-6 w-6 p-0 flex-shrink-0"
        aria-label={`Copy ${label || "text"}`}
      >
        {copied ? (
          <Check className="h-3 w-3 text-green-600" />
        ) : (
          <Copy className="h-3 w-3" />
        )}
      </Button>
    </div>
  );
}
