import React, { useState, useCallback } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import CodeViewer from "./CodeViewer";
import CopyCodeButton from "./CopyCodeButton";
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
    // Track which lines are highlighted in each panel
    const [leftHighlightedLines, setLeftHighlightedLines] = useState<number[]>(
        []
    );
    const [rightHighlightedLines, setRightHighlightedLines] = useState<
        number[]
    >([]);
    const [pythonHighlightedLines, setPythonHighlightedLines] = useState<
        number[]
    >([]);
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

    /**
     * Handles clicking on a line in either the left or right code panel
     * Highlights the line in the source panel and attempts to find and
     * highlight corresponding lines in the target panel and Python panel
     * @param lineNumber - The line number that was clicked
     * @param panel - Which panel was clicked ('left' or 'right')
     */
    const handlePanelLineClick = useCallback(
        (lineNumber: number, panel: 'left' | 'right') => {
            // Determine source and target based on which panel was clicked
            const isLeftPanel = panel === 'left';
            const sourceMapping = isLeftPanel ? leftPanel_data.sourceMapping : rightPanel_data.sourceMapping;
            const targetTitle = isLeftPanel ? rightPanel_data.title || "PTX" : leftPanel_data.title || "TTGIR";
            
            // Set highlight on the source panel first
            if (isLeftPanel) {
                setLeftHighlightedLines([lineNumber]);
            } else {
                setRightHighlightedLines([lineNumber]);
            }

            // Find corresponding lines in target panel using source mappings
            if (leftPanel.code && rightPanel.code) {
                const setTargetHighlightedLines = isLeftPanel ? setRightHighlightedLines : setLeftHighlightedLines;
                handleMappedLinesFound(
                    sourceMapping,
                    lineNumber,
                    setTargetHighlightedLines,
                    targetTitle
                );

            } else {
                // Clear target panel highlights if no mapping exists
                if (isLeftPanel) {
                    setRightHighlightedLines([]);
                } else {
                    setLeftHighlightedLines([]);
                }
            }

            // Find corresponding Python lines if Python source is shown
            if (showPythonSource && py_code_info) {
                const pythonLines = findPythonLines(
                    sourceMapping,
                    lineNumber
                );
                if (pythonLines.length > 0) {
                    setPythonHighlightedLines(pythonLines);
                } else {
                    setPythonHighlightedLines([]);
                }
            }

        },
        [
            leftPanel.code,
            rightPanel.code,
            leftPanel_data.title,
            rightPanel_data.title,
            leftPanel_data.sourceMapping,
            rightPanel_data.sourceMapping,
            handleMappedLinesFound,
            showPythonSource,
            py_code_info,
            findPythonLines
        ]
    );

    // Use curried functions to maintain the existing API for component props
    const handleLeftLineClick = useCallback(
        (lineNumber: number) => handlePanelLineClick(lineNumber, 'left'),
        [handlePanelLineClick]
    );

    const handleRightLineClick = useCallback(
        (lineNumber: number) => handlePanelLineClick(lineNumber, 'right'),
        [handlePanelLineClick]
    );

    /**
     * Handles clicking on a line in the Python source panel
     * Maps Python line to IR lines in both panels
     * @param lineNumber - The line number that was clicked in the Python panel
     */
    const handlePythonLineClick = useCallback(
        (lineNumber: number) => {
            setPythonHighlightedLines([lineNumber]);

            // compute the actual line number in the Python source code
            // Notice: lineNumber is 1-based, but py_start_line is 0-based
            // pythonLineNumber is 0-based
            const pythonLineNumber = lineNumber + py_start_line - 1 ;

            const highlightLines = (title: string, panel: 'left' | 'right') => {
                if (pythonMapping && pythonMapping[pythonLineNumber]) {
                    // get ir_type from title
                    const irType = getIRType(title);
                    const irLines = pythonMapping[pythonLineNumber][`${irType}_lines` as keyof SourceMapping] as number[] || [];

                    if (panel === 'left') {
                        // Ensure all values are numbers
                        setLeftHighlightedLines(irLines.map(Number));
                    } else {
                        // Ensure all values are numbers
                        setRightHighlightedLines(irLines.map(Number));
                    }
                } else {
                    panel === 'left' ? setLeftHighlightedLines([]) : setRightHighlightedLines([]);
                }
            };

            highlightLines(leftPanel_data.title, 'left');
            highlightLines(rightPanel_data.title, 'right');

        },
        [pythonMapping, py_start_line, leftPanel_data.title, rightPanel_data.title]
    );


    /**
     * A reusable code panel component
     */
    const CodePanel = React.memo<{
        title: string;
        displayLanguage: string;
        code: string;
        language: string;
        highlightedLines: number[];
        onLineClick: (lineNumber: number) => void;
        viewerId: string;
        otherViewerId?: string;
    }>(({
        title,
        displayLanguage,
        code,
        language,
        highlightedLines,
        onLineClick,
        viewerId,
        otherViewerId
    }) => (
        <div className="h-full">
            <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center">
                <span>{title}</span>
                <div className="flex items-center gap-2">
                    <span className="text-sm bg-blue-700 px-2 py-1 rounded">
                        {displayLanguage}
                    </span>
                    <CopyCodeButton 
                        code={code}
                        className="text-sm bg-blue-700 px-2 py-1 rounded" 
                    />
                </div>
            </div>
            <CodeViewer
                code={code}
                language={language}
                height="calc(100% - 40px)"
                highlightedLines={highlightedLines}
                onLineClick={onLineClick}
                theme="light"
                fontSize={14}
                viewerId={viewerId}
                otherViewerId={otherViewerId}
            />
        </div>
    ));

    // Render two or three panels based on whether Python source is available
    return (
        <div className="h-[calc(100vh-12rem)] bg-white rounded-lg overflow-hidden shadow border border-gray-200">
            {/* Resizable panel layout */}
            <PanelGroup direction="horizontal">
                {/* Left panel */}
                <Panel defaultSize={showPythonSource ? 33 : 50} minSize={20}>
                    <CodePanel
                        title={leftPanel_data.title}
                        displayLanguage={leftPanel_data.displayLanguage}
                        code={leftPanel_data.content}
                        language={leftPanel.language || "plaintext"}
                        highlightedLines={leftHighlightedLines}
                        onLineClick={handleLeftLineClick}
                        viewerId="left-panel"
                        otherViewerId="right-panel"
                    />
                </Panel>

                {/* Resize handle between panels */}
                <PanelResizeHandle className="w-2 bg-gray-200 hover:bg-gray-300 transition-colors" />

                {/* Right panel */}
                <Panel defaultSize={showPythonSource ? 33 : 50} minSize={20}>
                    <CodePanel
                        title={rightPanel_data.title}
                        displayLanguage={rightPanel_data.displayLanguage}
                        code={rightPanel_data.content}
                        language={rightPanel.language || "plaintext"}
                        highlightedLines={rightHighlightedLines}
                        onLineClick={handleRightLineClick}
                        viewerId="right-panel"
                        otherViewerId="left-panel"
                    />
                </Panel>
                {/* Python source panel (only shown if Python source is available) */}
                {showPythonSource && (
                    <>
                        <PanelResizeHandle className="w-2 bg-gray-200 hover:bg-gray-300 transition-colors" />
                        <Panel defaultSize={33} minSize={20}>
                            <CodePanel
                                title="Python Source"
                                displayLanguage={getDisplayLanguage("python")}
                                code={py_code}
                                language="python"
                                highlightedLines={pythonHighlightedLines}
                                onLineClick={handlePythonLineClick}
                                viewerId="python-panel"
                            />
                        </Panel>
                    </>
                )}
            </PanelGroup>
        </div>
    );
};

export default CodeComparisonView;
