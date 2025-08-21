/**
 * A normalized set of tensor metadata fields we may extract from trace arguments.
 */
export interface TensorMetadata {
    /** Data type, e.g. torch.float32 */
    dtype?: string;
    /** Tensor shape, rendered as [d0, d1, ...] */
    shape?: number[] | string;
    /** Tensor strides, rendered as [s0, s1, ...] */
    strides?: number[] | string;
    /** Device string, e.g. cuda:0 */
    device?: string;
    /** Optional element count or logical length */
    length?: number;
    /** Optional layout string (if provided by trace) */
    layout?: string;
    /** Whether tensor is contiguous in memory */
    contiguous?: boolean;
    /** Total bytes occupied by the tensor */
    nbytes?: number;
    /** Autograd flag (if surfaced) */
    requires_grad?: boolean;
    /** Storage offset value (if surfaced) */
    storage_offset?: number;
    /** Allow unknown additional tensor attributes without breaking type checks */
    [key: string]: any;
}

/**
 * Determines whether an extracted argument represents a tensor.
 * Only checks for a string type equal to 'tensor' (case-insensitive).
 */
export function isTensorArg(arg: any): boolean {
    if (!arg || typeof arg.type !== 'string') return false;
    return arg.type.trim().toLowerCase() === 'tensor';
}

/**
 * Extracts known tensor metadata fields from an argument. The function looks in two places:
 * 1) The argument object itself (top-level)
 * 2) The `value` field when present (nested)
 * The first defined occurrence of a field wins.
 */
export function extractTensorMetadata(arg: any): TensorMetadata {
    const meta: TensorMetadata = {};
    const sources = [arg ?? {}, arg?.value ?? {}];

    // Safe lookup of a property across multiple source objects
    const pick = (key: string) => {
        for (const src of sources) {
            if (src && Object.prototype.hasOwnProperty.call(src, key)) {
                const value = src[key];
                if (value !== undefined) return value;
            }
        }
        return undefined;
    };

    // Map target metadata fields to potential source keys
    const keyMap: Array<[keyof TensorMetadata, string]> = [
        ['dtype', 'dtype'],
        ['shape', 'shape'],
        ['strides', 'strides'],
        ['device', 'device'],
        ['length', 'length'],
        ['layout', 'layout'],
        ['contiguous', 'contiguous'],
        ['nbytes', 'nbytes'],
        ['requires_grad', 'requires_grad'],
        ['storage_offset', 'storage_offset'],
    ];

    for (const [targetKey, sourceKey] of keyMap) {
        const value = pick(sourceKey);
        if (value !== undefined) (meta as any)[targetKey] = value;
    }

    return meta;
}

/**
 * Formats a compact, labeled, single-line tensor summary from metadata.
 * Missing fields are omitted; if all are missing, a dash ('-') is returned.
 */
export function formatTensorSummary(meta: TensorMetadata): string {
    const parts: string[] = [];

    const render = (value: any): string => Array.isArray(value)
        ? `[${value.join(', ')}]`
        : String(value);

    const add = (label: string, value: any) => {
        if (value !== undefined) parts.push(`${label}: ${render(value)}`);
    };

    add('dtype', meta.dtype);
    add('shape', meta.shape);
    add('strides', meta.strides);
    add('device', meta.device);
    add('len', meta.length);
    add('layout', meta.layout);
    add('contiguous', meta.contiguous);
    add('nbytes', meta.nbytes);
    add('storage_offset', meta.storage_offset);

    return parts.length > 0 ? parts.join(', ') : '-';
}


