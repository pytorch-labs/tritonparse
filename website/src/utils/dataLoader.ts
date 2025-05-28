/**
 * Source mapping information that connects lines in IR code to source code
 */
export interface SourceMapping {
    line: number;
    file?: string;
    column?: number;
    // The {ir_type}_line fields are the line numbers in the current IR file that corresponds to
    // the current line in the source code. It should be same with the key in the source_mapping.
    ttgir_line?: number;
    ttir_line?: number;
    ptx_line?: number;
    amdgcn_line?: number;
    llir_line?: number;
    ptx_lines?: number[]; // Array of corresponding PTX lines
    ttir_lines?: number[]; // Array of corresponding TTIR lines
    ttgir_lines?: number[]; // Array of corresponding TTGIR lines
    llir_lines?: number[]; // Array of corresponding LLIR lines
    amdgcn_lines?: number[]; // Array of corresponding AMDGCN lines
}

/**
 * Get IR type from file name
 * @param fileName - The name of the file
 * @returns The type of IR file (without the dot)
 */
export function getIRType(fileName: string): string {
    // 1. Extract the file extension
    const extMatch = fileName.toLowerCase().match(/\.([^.]+)$/);
    if (extMatch) return extMatch[1];

    // 2. If there is no extension or the format does not match, return the original value
    return fileName.toLowerCase();
}

/**
 * Represents an IR file with content and source mapping information
 */
export interface IRFile {
    content: string;
    source_mapping?: Record<string, SourceMapping>;
}


/**
 * Represents a stack trace entry with line, function name, file info
 */
export interface StackEntry {
    line: number;
    name: string;
    filename: string | number | [string, number]; // Can be index, array [filepath, index], or filepath string
    loc: string;
}

/**
 * Represents kernel compilation metadata
 */
export interface KernelMetadata {
    hash?: string;
    target?: {
        backend?: string;
        arch?: number;
        warp_size?: number;
    };
    num_warps?: number;
    num_ctas?: number;
    num_stages?: number;
    maxnreg?: number | null;
    cluster_dims?: number[];
    ptx_version?: number | null;
    enable_fp_fusion?: boolean;
    launch_cooperative_grid?: boolean;
    supported_fp8_dtypes?: string[];
    [key: string]: any; // For other metadata properties
}

/**
 * Python source code information
 */
export interface PythonSourceCodeInfo {
    file_path: string;
    start_line: number;
    end_line?: number; // End line number (inclusive)
    code?: string;
}

/**
 * Raw log entry from the Triton trace log
 */
export interface LogEntry {
    event_type: string;
    pid?: number;
    stack?: StackEntry[];
    timestamp?: string; // Format: "2025-03-25T13:22:04.%fZ"
    payload?: {
        metadata?: KernelMetadata;
        file_path?: Record<string, string>; // Mapping from filename to filepath
        file_content?: Record<string, string>; // Mapping from filename to content
        source_mappings?: Record<string, Record<string, SourceMapping>>; // Alternative field name for source_mapping
        python_source?: PythonSourceCodeInfo;
    };
}

/**
 * Processed kernel data structure for rendering in the UI
 */
export interface ProcessedKernel {
    name: string; // Inferred from filename
    sourceFiles: string[]; // All related source files
    stack: StackEntry[];
    irFiles: Record<string, string>; // IR file contents
    filePaths: Record<string, string>; // IR file paths
    sourceMappings?: Record<string, Record<string, SourceMapping>>; // Source mappings for each IR file
    pythonSourceInfo?: PythonSourceCodeInfo; // Python source code information
    metadata?: KernelMetadata; // Compilation metadata
}

/**
 * Parses NDJSON text data (Newline Delimited JSON)
 * @param textData - The NDJSON text data to parse
 * @returns Array of LogEntry objects
 */
export function parseLogData(textData: string): LogEntry[] {
    if (typeof textData !== 'string') {
        throw new Error("Input must be a string in NDJSON format");
    }

    try {
        const lines = textData.split('\n').filter((line: string) => line.trim() !== '');
        const entries: LogEntry[] = [];

        for (const line of lines) {
            try {
                const parsedLine = JSON.parse(line);
                if (parsedLine && typeof parsedLine === 'object') {
                    entries.push(parsedLine);
                }
            } catch (e) {
                console.warn(`Failed to parse line as JSON: ${line.substring(0, 100)}...`);
                // Continue processing other lines even if one fails
            }
        }

        if (entries.length === 0) {
            throw new Error("No valid JSON entries found in NDJSON data");
        }

        return entries;
    } catch (error) {
        console.error("Error parsing NDJSON data:", error);
        throw error;
    }
}

/**
 * Detects if a file is in gzip format by checking its header bytes
 * @param buffer - ArrayBuffer containing the file data
 * @returns Boolean indicating if the file is in gzip format
 */
function isGzipFile(buffer: ArrayBuffer): boolean {
    // Check for gzip magic number (first two bytes should be 0x1F, 0x8B)
    const header = new Uint8Array(buffer.slice(0, 2));
    return header[0] === 0x1F && header[1] === 0x8B;
}

/**
 * Decompresses a gzip file using CompressionStream API
 * @param buffer - ArrayBuffer containing the gzip data
 * @returns Promise resolving to decompressed text
 */
async function decompressGzip(buffer: ArrayBuffer): Promise<string> {
    try {
        // Check if CompressionStream is supported
        if (!('DecompressionStream' in window)) {
            throw new Error('DecompressionStream API is not supported in this browser');
        }

        // Create a decompression stream
        const ds = new DecompressionStream('gzip');
        const decompressedStream = new Blob([buffer]).stream().pipeThrough(ds);
        const decompressedBlob = await new Response(decompressedStream).blob();
        return await decompressedBlob.text();
    } catch (error) {
        console.error('Error decompressing gzip file:', error);
        throw new Error(`Failed to decompress gzip file: ${error instanceof Error ? error.message : String(error)}`);
    }
}

/**
 * Processes ArrayBuffer data, handling gzip decompression if needed
 * @param buffer - ArrayBuffer containing the data
 * @returns Promise resolving to text string
 */
async function processArrayBuffer(buffer: ArrayBuffer): Promise<string> {
    // Check if file is gzip compressed
    if (isGzipFile(buffer)) {
        try {
            const textData = await decompressGzip(buffer);
            return textData;
        } catch (error) {
            console.error("Error decompressing gzip data:", error);
            throw new Error(`Failed to decompress gzip data: ${error instanceof Error ? error.message : String(error)}`);
        }
    } else {
        // Convert ArrayBuffer to string if it's not compressed
        const decoder = new TextDecoder();
        const textData = decoder.decode(buffer);
        return textData;
    }
}

/**
 * Loads log data from a URL and parses it as NDJSON
 * @param url - The URL of the log file to load
 * @returns Promise resolving to an array of LogEntry objects
 */
export async function loadLogData(url: string): Promise<LogEntry[]> {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load log data: ${response.statusText}`);
        }

        const buffer = await response.arrayBuffer();
        const textData = await processArrayBuffer(buffer);

        return parseLogData(textData);
    } catch (error) {
        console.error("Error loading log data:", error);
        throw error;
    }
}

/**
 * Process raw log entries to extract kernel information
 * @param logEntries - Array of log entries from the trace file
 * @returns Array of processed kernel objects ready for display
 */
export function processKernelData(logEntries: LogEntry[]): ProcessedKernel[] {
    const kernels: ProcessedKernel[] = [];
    for (let i = 0; i < logEntries.length; i++) {
        const entry = logEntries[i];
        // Check for kernel events by event_type
        if (entry.event_type === "compilation" && entry.payload) {
            // Ensure payload has file_path and file_content
            if (!entry.payload.file_path || !entry.payload.file_content) {
                console.warn(
                    "Kernel event missing file_path or file_content",
                    entry.payload
                );
                continue;
            }
            // Extract kernel name from IR filename
            const irFileNames = Object.keys(entry.payload.file_path);
            let kernelName = "unknown_kernel";
            // Use first IR file name to determine kernel name
            if (irFileNames.length > 0) {
                const fileName = irFileNames[0];
                const nameParts = fileName.split(".");
                kernelName =
                    nameParts.length > 1
                        ? nameParts.slice(0, -1).join(".")
                        : fileName;
            }

            // Extract source mapping information from payload if available
            let sourceMappings: Record<
                string,
                Record<string, SourceMapping>
            > = {};

            if (entry.payload.source_mappings) {
                // Use source mappings from the trace file
                sourceMappings = entry.payload.source_mappings;
            }

            // Create processed kernel object and add to results
            kernels.push({
                name: kernelName,
                sourceFiles: entry.stack?.map(entry =>
                    typeof entry.filename === 'string' ? entry.filename :
                    Array.isArray(entry.filename) ? entry.filename[0] : "unknown"
                ) || [],
                stack: entry.stack || [],
                irFiles: entry.payload.file_content,
                filePaths: entry.payload.file_path,
                sourceMappings,
                pythonSourceInfo: entry.payload.python_source,
                metadata: entry.payload.metadata,
            });

        }
    }
    return kernels;
}
