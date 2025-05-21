import React, { useState, useCallback } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import CodeViewer from "./CodeViewer";
import {
    IRFile,
    PythonSourceCodeInfo,
    SourceMapping,
    getIRType,
} from "../utils/dataLoader";
import { getDisplayLanguage } from "./TritonIRs";

/**
 * Props for a single code panel
 */
interface PanelProps {
    code?: IRFile;     // IR file with source mapping
    content?: string;  // Direct content (alternative to code)
    language?: string; // Language for syntax highlighting
    title?: string;    // Title for the panel
}

/**
 * Props for the CodeComparisonView component
 */
interface CodeComparisonViewProps {
    leftPanel: PanelProps;  // Left panel properties
    rightPanel: PanelProps; // Right panel properties
    py_code_info?: PythonSourceCodeInfo; // Python source code information
    showPythonSource?: boolean; // Flag to show/hide Python source panel
    pythonMapping?: Record<string, SourceMapping>; // Python source code to all IR file mappings
}

/**
 * CodeComparisonView component that renders two or three code panels side by side
 * with optional line highlighting and synchronization between panels
 */
const CodeComparisonView: React.FC<CodeComparisonViewProps> = ({
    leftPanel,
    rightPanel,
    py_code_info,
    showPythonSource = false,
    pythonMapping,
}) => {
    /**
     * Process panel properties to get content, source mapping and display language
     * @param panel Panel properties
     * @param defaultTitle Default title to use if panel title is undefined
     * @returns Processed panel data
     */
    const processPanelProps = (panel: PanelProps, defaultTitle: string) => {
        const title = panel.title || defaultTitle;
        const content = panel.content || (panel.code ? panel.code.content : "");
        const sourceMapping = panel.code?.source_mapping || {};
        const displayLanguage = getDisplayLanguage(title);

        return { title, content, sourceMapping, displayLanguage };
    };
    // Process panel properties
    const leftPanel_data = processPanelProps(leftPanel, "TTGIR");
    const rightPanel_data = processPanelProps(rightPanel, "PTX");

    // Get Python source code info
    const py_code = py_code_info?.code || "";
    const py_file_path = py_code_info?.file_path || "";
    const py_start_line = py_code_info?.start_line || 0;

    /**
     * Find Python lines that correspond to the source location in the IR file
     * @param sourceMapping Source mapping from IR file
     * @param lineNumber Line number in the IR file
     * @returns Array of Python line numbers that correspond to the source location
     */
    const findPythonLines = useCallback(
        (
            sourceMapping: Record<string, SourceMapping>,
            lineNumber: number
        ): number[] => {
            if (!sourceMapping || !py_code_info || !py_code) return [];

            const lineKey = lineNumber.toString();
            const mapping = sourceMapping[lineKey];

            if (!mapping) return [];

            // If we have a direct line mapping to Python source
            if (mapping.file && mapping.line) {
                // Check if the file path in the mapping matches the Python source file
                if (mapping.file.includes(py_file_path)) {
                    // Adjust for the start line in Python source
                    // The source line in the mapping is absolute, but our display is relative to start_line
                    const adjustedLine = Number(mapping.line) - py_start_line + 1;
                    // Get the actual line content to verify
                    const pythonLines = py_code.split("\n");
                    if (
                        adjustedLine >= 1 &&
                        adjustedLine <= pythonLines.length
                    ) {
                        // Return the adjusted line as a number
                        return [Number(adjustedLine)];
                    } else {
                        console.error(
                            `Adjusted line ${adjustedLine} is out of range (1-${pythonLines.length})`
                        );
                    }
                }
            }

            return [];
        },
        [py_code_info, py_code]
    );

    /**
     * Handles finding and highlighting mapped lines when a line is clicked in either panel
     * @param sourceMappings - Source mapping information for the source panel
     * @param targetMappings - Source mapping information for the target panel
     * @param lineNumber - The line number that was clicked
     * @param setTargetHighlightedLines - Function to set highlighted lines in target panel
     * @param targetTitle - The title of the target panel (to determine which mapping array to use)
     */
    const handleMappedLinesFound = useCallback(
        (
            sourceMappings: Record<string, SourceMapping>,
            targetMappings: Record<string, SourceMapping>,
            lineNumber: number,
            setTargetHighlightedLines: (lines: number[]) => void,
            targetTitle: string
        ) => {
            const lineKey = lineNumber.toString();
            // Skip if no source mapping exists for this line
            if (!sourceMappings[lineKey]) {
                setTargetHighlightedLines([]); // Ensure we clear highlights when no mapping exists
                return;
            }

            const sourceMapping = sourceMappings[lineKey];
            // Initialize array to hold matched lines
            let mappedLines: number[] = [];
            const targetIRType = getIRType(targetTitle);
            // Define the possible IR types and their corresponding property names
            const irTypesToCheck = [
                { type: "ttgir", property: "ttgir_lines" },
                { type: "ttir", property: "ttir_lines" },
                { type: "ptx", property: "ptx_lines" },
                { type: "llir", property: "llir_lines" },
                { type: "amdgcn", property: "amdgcn_lines" }
            ];

            // Loop through IR types to find matching lines
            let found = false;
            for (const { type, property } of irTypesToCheck) {
                if (targetIRType === type && sourceMapping[property as keyof SourceMapping] !== undefined) {
                    mappedLines = sourceMapping[property as keyof SourceMapping] as number[];
                    found = true;
                    break;
                }
            }

            // If no matching IR type was found
            if (!found) {
                setTargetHighlightedLines([]);
                return;
            }

            if (mappedLines && mappedLines.length > 0) {
                // ensure all line numbers are numbers (not strings)
                const numericMappedLines = mappedLines.map((line) => {
                    if (typeof line === "string") {
                        return parseInt(line, 10);
                    }
                    return line;
                });

                // Force array to ensure it's not undefined or null and ensure numeric values
                setTargetHighlightedLines(numericMappedLines.map(Number));

            } else {
                setTargetHighlightedLines([]);
            }
        },
        [leftPanel_data.content, rightPanel_data.content]
    );



    // Render two or three panels based on whether Python source is available
    return (
        <div className="h-[calc(100vh-12rem)] bg-white rounded-lg overflow-hidden shadow border border-gray-200">
            {/* Resizable panel layout */}
            <PanelGroup direction="horizontal">

            </PanelGroup>
        </div>
    );
};

export default CodeComparisonView;
