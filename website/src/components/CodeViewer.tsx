import React, { useState, useEffect, useRef, useCallback } from "react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneLight,
  oneDark,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import type { SourceMapping } from "../utils/dataLoader";

// Import language support
import llvm from 'react-syntax-highlighter/dist/esm/languages/prism/llvm';
import c from 'react-syntax-highlighter/dist/esm/languages/prism/c';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';

// Register languages with the syntax highlighter
SyntaxHighlighter.registerLanguage('llvm', llvm);
SyntaxHighlighter.registerLanguage('c', c);
SyntaxHighlighter.registerLanguage('python', python);

/**
 * Props for the CodeViewer component
 */
interface CodeViewerProps {
  code: string; // The source code to display
  language?: string; // The programming language for syntax highlighting
  height?: string; // Height of the code viewer container
  theme?: "light" | "dark"; // Color theme
  fontSize?: number; // Font size for the code
  highlightedLines?: number[]; // Array of line numbers to highlight
  onLineClick?: (lineNumber: number) => void; // Callback when a line is clicked
  viewerId?: string; // Unique identifier for this viewer (used for mapping)
  otherViewerId?: string; // Identifier for the paired viewer (for mapping)
  sourceMapping?: Record<string, SourceMapping>; // Source mapping information
  onMappedLinesFound?: (mappedLines: number[]) => void; // Callback when mapped lines are found
}

/**
 * Maps our internal language names to syntax highlighter languages
 * @param language Internal language identifier
 * @returns Syntax highlighter language identifier
 */
const mapLanguageToHighlighter = (language: string): string => {
  switch (language.toLowerCase()) {
    case 'mlir':
      return 'mlir';
    case 'llvm':
      return 'llvm';
    case 'ptx':
      return 'ptx';
    case 'python':
      return 'python';
    default:
      return 'plaintext';
  }
};

/**
 * Split code into lines
 * @param code The code content as string
 * @returns Array of code lines
 */
const splitIntoLines = (code: string): string[] => {
  return code.split('\n');
};

/**
 * Creates a debounced function that delays invoking func until after wait milliseconds
 * @param func The function to debounce
 * @param wait The number of milliseconds to delay
 * @returns Debounced function
 */
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Basic code viewer using pre/code tags for very large files
 * Disables syntax highlighting for extremely large files to maintain performance
 */
const BasicCodeViewer: React.FC<CodeViewerProps> = ({
  code,
  height = "100%",
  fontSize = 14,
  highlightedLines = [],
  onLineClick,
  viewerId,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const lines = splitIntoLines(code);

  const handleLineClick = useCallback((lineNumber: number) => {
    if (onLineClick) {
      onLineClick(lineNumber);
    }
  }, [onLineClick, viewerId]);

  return (
    <div
      ref={containerRef}
      style={{
        height,
        overflowY: "auto",
        fontSize: `${fontSize}px`,
        backgroundColor: "#fff"
      }}
      className={`code-viewer ${highlightedLines.length > 0 ? 'has-highlights' : ''}`}
      data-viewer-id={viewerId}
      data-highlighted-lines={highlightedLines.join(',')}
    >
      <pre style={{
        margin: 0,
        padding: "0.5em",
        fontFamily: "Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace"
      }}>
        <code>
          {lines.map((line, index) => {
            const lineNumber = index + 1;
            const isHighlighted = highlightedLines.includes(lineNumber);

            return (
              <div
                key={index}
                style={{
                  paddingLeft: "3.8em",
                  position: "relative",
                  whiteSpace: "pre-wrap",
                  backgroundColor: isHighlighted
                    ? "rgba(255, 215, 0, 0.4)" // More golden yellow
                    : "transparent",
                  borderLeft: isHighlighted ? "3px solid orange" : "none",
                  cursor: onLineClick ? "pointer" : "text"
                }}
                onClick={() => handleLineClick(lineNumber)}
                data-line-number={lineNumber}
                className={isHighlighted ? 'highlighted-line' : ''}
              >
                <span style={{
                  position: "absolute",
                  left: 0,
                  userSelect: "none",
                  opacity: 0.5,
                  width: "3em",
                  textAlign: "right",
                  paddingRight: "0.5em"
                }}>
                  {lineNumber}
                </span>
                {line || " "}
              </div>
            );
          })}
        </code>
      </pre>
    </div>
  );
};


/**
 * Standard code viewer for smaller files
 */
const StandardCodeViewer: React.FC<CodeViewerProps> = ({
  code,
  language = "plaintext",
  height = "100%",
  theme = "light",
  fontSize = 14,
  highlightedLines = [],
  onLineClick,
  viewerId,
}) => {
  // Map the language to the appropriate highlighter language
  const highlighterLanguage = mapLanguageToHighlighter(language);

  // Enhanced line click handler that processes source mapping
  const handleLineClick = useCallback((lineNumber: number) => {
    if (onLineClick) {
      onLineClick(lineNumber);
    }
  }, [onLineClick, viewerId]);

  /**
   * Configures styling and behavior for each line of code
   * Creates custom styling for highlighted lines and handles click events
   * @param lineNumber - The line number to configure properties for
   * @returns Props object with style and click handler
   */
  const lineProps = (lineNumber: number) => {
    // Create styles for the line
    const style: React.CSSProperties = {
      display: "block",
      cursor: onLineClick ? "pointer" : "text",
    };

    // Apply background color if this line should be highlighted
    const isHighlighted = highlightedLines.includes(lineNumber);
    if (isHighlighted) {
      // Use a more vibrant highlight color with better contrast
      style.backgroundColor = theme === "light"
        ? "rgba(255, 215, 0, 0.4)" // More golden yellow for light theme
        : "rgba(255, 215, 0, 0.3)"; // Similar but slightly dimmer for dark theme
      style.borderLeft = "3px solid orange"; // Add left border for better visibility
      style.paddingLeft = "6px"; // Add some padding to offset the border
    }

    return {
      style,
      onClick: () => handleLineClick(lineNumber),
    };
  };

  return (
    <div
      style={{ height, overflowY: "auto", fontSize: `${fontSize}px` }}
      className={`code-viewer ${highlightedLines.length > 0 ? 'has-highlights' : ''}`}
      data-viewer-id={viewerId}
      data-highlighted-lines={highlightedLines.join(',')}
    >
      <SyntaxHighlighter
        language={highlighterLanguage}
        style={theme === "light" ? oneLight : oneDark}
        showLineNumbers
        wrapLines
        lineProps={lineProps}
        customStyle={{
          margin: 0,
          fontSize: "inherit",
          height: "100%",
          backgroundColor: theme === "light" ? "#fff" : "#1E1E1E",
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

/**
 * CodeViewer component that renders code with syntax highlighting
 * and supports line highlighting and clicking.
 * Automatically chooses between standard, optimized, or basic viewer based on code size.
 */
const CodeViewer: React.FC<CodeViewerProps> = (props) => {
  // Add debug log to monitor props changes
  useEffect(() => {

    // Add inline style for highlighted lines to ensure they're visible
    if (props.highlightedLines && props.highlightedLines.length > 0) {
      const styleId = `highlight-style-${props.viewerId || 'default'}`;
      let styleEl = document.getElementById(styleId) as HTMLStyleElement;

      if (!styleEl) {
        styleEl = document.createElement('style');
        styleEl.id = styleId;
        document.head.appendChild(styleEl);
      }

      // Create a style rule for highlighted lines
      const lines = props.highlightedLines.join(', .line-');
      styleEl.innerHTML = `
        .highlighted-line, .line-${lines} {
          background-color: rgba(255, 215, 0, 0.4) !important;
          border-left: 3px solid orange !important;
        }
      `;

    }
  }, [props.highlightedLines, props.viewerId]);

  // Check if there's any code at all
  if (!props.code) {
    return (
      <div className="code-viewer" style={{
        height: props.height || "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: props.theme === "light" ? "#fff" : "#1E1E1E",
        color: props.theme === "light" ? "#333" : "#eee",
      }}>
        No code to display
      </div>
    );
  }

  // Use standard viewer for smaller files
  return <StandardCodeViewer {...props} />;
};

// Use React.memo to prevent unnecessary re-renders
export default React.memo(CodeViewer);
