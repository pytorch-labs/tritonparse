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
 * Represents an IR file with content and source mapping information
 */
export interface IRFile {
    content: string;
    source_mapping?: Record<string, SourceMapping>;
}
