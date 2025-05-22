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
