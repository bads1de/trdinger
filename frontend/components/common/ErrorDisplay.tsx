import React from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertCircle, RefreshCw } from "lucide-react";

interface ErrorDisplayProps {
  message: string;
  onRetry?: () => void;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ message, onRetry }) => {
  if (!message) {
    return null;
  }

  return (
    <Alert
      variant="destructive"
      className="flex flex-col items-center justify-center p-6 bg-red-900/20 border-red-500/50"
    >
      <div className="flex items-center mb-4">
        <AlertCircle className="h-6 w-6 mr-3" />
        <AlertDescription className="text-lg font-semibold">
          {message}
        </AlertDescription>
      </div>
      {onRetry && (
        <Button
          onClick={onRetry}
          variant="outline"
          className="mt-4 bg-transparent border-red-400 text-red-400 hover:bg-red-400 hover:text-gray-900"
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          再試行
        </Button>
      )}
    </Alert>
  );
};

export default ErrorDisplay;
