/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 */

import React, { useState } from "react";
import CodeViewer from "./CodeViewer";
import { IRFile } from "../utils/dataLoader";
import { getDisplayLanguage } from "./TritonIRs";

/**
 * Props for the SingleCodeViewer component
 */
interface SingleCodeViewerProps {
  irFile?: IRFile; // IR file object containing content and source mappings
  irContent?: string; // Direct code content as string (alternative to irFile)
  title: string; // Title to display for the code view
  language?: string; // Language for syntax highlighting
  onBack: () => void; // Callback function when back button is clicked
}

/**
 * SingleCodeViewer component that displays a single IR file with syntax highlighting
 * Used for detailed viewing of a specific IR file
 */
const SingleCodeViewer: React.FC<SingleCodeViewerProps> = ({
  irFile,
  irContent,
  title,
  language = "plaintext",
  onBack,
}) => {
  // Track highlighted lines for self-referential mapping
  const [highlightedLines, setHighlightedLines] = useState<number[]>([]);

  // Determine content to display (either from direct content or from IRFile)
  const codeContent = irContent || (irFile ? irFile.content : "");
  const displayLanguage = getDisplayLanguage(title);

  // Get source mapping if available
  const sourceMapping = irFile?.source_mapping;

  /**
   * Handle line click within a single file view
   * Can be used to highlight related lines within the same file
   */
  const handleLineClick = (lineNumber: number) => {
    // Toggle highlight for the clicked line
    setHighlightedLines([lineNumber]);

    // If there's source mapping available, we could highlight related lines
    // For example lines that map to the same source code location
    if (sourceMapping) {
      const lineKey = lineNumber.toString();
      const clickedMapping = sourceMapping[lineKey];

      if (clickedMapping && clickedMapping.ttgir_line) {
        // Find all lines that map to the same TTGIR line
        const relatedLines = Object.entries(sourceMapping)
          .filter(
            ([_, mapping]) =>
              mapping.ttgir_line === clickedMapping.ttgir_line &&
              parseInt(lineKey, 10) !== parseInt(_, 10) // Skip the clicked line itself
          )
          .map(([line, _]) => parseInt(line, 10));

        if (relatedLines.length > 0) {
          // Include the clicked line and any related lines
          setHighlightedLines([lineNumber, ...relatedLines]);
        }
      }
    }
  };

  /**
   * Handle finding mapped lines
   */
  const handleMappedLinesFound = (mappedLines: number[]) => {
    if (mappedLines.length > 0) {
      setHighlightedLines(prev => {
        // Filter out any duplicates
        const combined = [...prev, ...mappedLines];
        return Array.from(new Set(combined));
      });
    }
  };

  return (
    <div className="p-6">
      {/* Header with back button and title */}
      <div className="flex items-center mb-4">
        <button
          onClick={onBack}
          className="text-blue-600 hover:text-blue-800 flex items-center mr-4"
        >
          {/* Back arrow SVG icon */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 mr-1"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z"
              clipRule="evenodd"
            />
          </svg>
          Back
        </button>
        <div>
          <h1 className="text-2xl font-bold text-gray-800">{title}</h1>
          <p className="text-gray-600">Language: {displayLanguage}</p>
        </div>
      </div>

      {/* Code viewer container */}
      <div className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden">
        {/* Panel title bar */}
        <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center">
          <span>{title}</span>
          <span className="text-sm bg-blue-700 px-2 py-1 rounded">
            {displayLanguage}
          </span>
        </div>
        {/* Code content area with fixed height */}
        <div className="h-[calc(100vh-12rem)]">
          <CodeViewer
            code={codeContent}
            language={language}
            height="100%"
            theme="light"
            fontSize={16}
            highlightedLines={highlightedLines}
            onLineClick={handleLineClick}
            sourceMapping={sourceMapping}
            onMappedLinesFound={handleMappedLinesFound}
            viewerId="single-viewer"
          />
        </div>
      </div>
    </div>
  );
};

export default SingleCodeViewer;
