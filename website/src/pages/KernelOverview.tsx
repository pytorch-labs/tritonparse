import React from "react";
import { ProcessedKernel } from "../utils/dataLoader";

interface KernelOverviewProps {
  kernels: ProcessedKernel[];
  selectedKernel: number;
  onSelectKernel: (index: number) => void;
  onViewIR: (irType: string) => void;
}

/**
 * Formats a value for display in the metadata section
 * @param value - The value to format
 * @returns Formatted string representation
 */
const formatMetadataValue = (value: any): string => {
  if (value === null) {
    return "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (Array.isArray(value)) {
    return JSON.stringify(value);
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
};

/**
 * Component for displaying a single metadata item with consistent styling
 */
interface MetadataItemProps {
  label: string;
  value: React.ReactNode;
  span?: number; // Number of columns to span (default: 1)
}

const MetadataItem: React.FC<MetadataItemProps> = ({
  label,
  value,
  span = 1
}) => (
  <div className={`flex flex-col ${span > 1 ? `col-span-${span}` : ''}`}>
    <span className="text-sm font-medium text-gray-500">
      {label}
    </span>
    <span className="font-mono text-sm overflow-hidden text-ellipsis">
      {value}
    </span>
  </div>
);

/**
 * Gets the actual file path from a stack entry's filename
 * @param entry - The stack entry
 */
const getSourceFilePath = (entry: any): string => {
  if (typeof entry.filename === "string") {
    return entry.filename;
  }
  return "Invalid filename format";
};

const KernelOverview: React.FC<KernelOverviewProps> = ({
  kernels,
  selectedKernel,
  onSelectKernel,
  onViewIR,
}) => {
  if (kernels.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-800">No kernel data available</div>
      </div>
    );
  }

  const kernel = kernels[selectedKernel];

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">
        Triton Kernel Overview
      </h1>

      {/* Kernel Selection */}
      <div className="bg-white rounded-lg p-4 mb-4 shadow border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Available Kernels
        </h2>
        <div className="flex flex-wrap gap-2">
          {kernels.map((k, index) => (
            <button
              key={index}
              className={`px-4 py-2 text-sm rounded-md transition-colors whitespace-nowrap ${
                index === selectedKernel
                  ? "bg-blue-100 border border-blue-300 text-blue-800"
                  : "bg-gray-50 border border-gray-200 hover:bg-blue-50 text-gray-800"
              }`}
              onClick={() => onSelectKernel(index)}
            >
              <div className="font-medium">{k.name}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Kernel Details */}
      <div className="bg-white rounded-lg p-4 mb-4 shadow border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Kernel Details: {kernel.name}
        </h2>

        {/* Metadata Section */}
        {kernel.metadata && (
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-3 text-gray-800">
              Compilation Metadata
            </h3>
            <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* Hash */}
                {kernel.metadata.hash && (
                  <MetadataItem label="Hash" value={kernel.metadata.hash} />
                )}

                {/* Target Info */}
                {kernel.metadata.target && (
                  <>
                    <MetadataItem label="Backend" value={kernel.metadata.target.backend || "N/A"} />
                    <MetadataItem label="Architecture" value={kernel.metadata.target.arch ? `SM ${kernel.metadata.target.arch}` : "N/A"} />
                    <MetadataItem label="Warp Size" value={kernel.metadata.target.warp_size || "N/A"} />
                  </>
                )}

                {/* Execution Configuration */}
                <MetadataItem label="Num Warps" value={kernel.metadata.num_warps !== undefined ? kernel.metadata.num_warps : "N/A"} />
                <MetadataItem label="Num CTAs" value={kernel.metadata.num_ctas !== undefined ? kernel.metadata.num_ctas : "N/A"} />
                <MetadataItem label="Num Stages" value={kernel.metadata.num_stages !== undefined ? kernel.metadata.num_stages : "N/A"} />

                {/* Cluster Dimensions */}
                {kernel.metadata.cluster_dims && (
                  <MetadataItem label="Cluster Dimensions" value={kernel.metadata.cluster_dims.join(" × ")} />
                )}

                {/* Other Metadata */}
                <MetadataItem label="FP Fusion" value={kernel.metadata.enable_fp_fusion !== undefined ? kernel.metadata.enable_fp_fusion ? "Enabled" : "Disabled" : "N/A"} />
                <MetadataItem label="Cooperative Grid" value={kernel.metadata.launch_cooperative_grid !== undefined ? kernel.metadata.launch_cooperative_grid ? "Yes" : "No" : "N/A"} />

                {/* Supported FP8 Types */}
                {kernel.metadata.supported_fp8_dtypes &&
                  kernel.metadata.supported_fp8_dtypes.length > 0 && (
                    <MetadataItem label="Supported FP8 Types" value={kernel.metadata.supported_fp8_dtypes.join(", ")} span={2} />
                  )}

                {/* Additional metadata fields */}
                {Object.entries(kernel.metadata)
                  .filter(
                    ([key]) =>
                      ![
                        "hash",
                        "target",
                        "num_warps",
                        "num_ctas",
                        "num_stages",
                        "cluster_dims",
                        "enable_fp_fusion",
                        "launch_cooperative_grid",
                        "supported_fp8_dtypes",
                      ].includes(key)
                  )
                  .map(([key, value]) => (
                    <MetadataItem key={key} label={key.split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ")} value={formatMetadataValue(value)} />
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* Stack Trace */}
        <div className="mb-4">
          <h3 className="text-lg font-medium mb-2 text-gray-800">
            Stack Trace
          </h3>
          <div className="bg-gray-50 p-3 rounded-md border border-gray-200 overflow-auto max-h-64">
            {kernel.stack.map((entry, index) => (
              <div key={index} className="mb-1 font-mono text-sm">
                <span className="text-blue-600">
                  {getSourceFilePath(entry)}
                </span>
                :<span className="text-red-600">{entry.line}</span> -
                <span className="text-green-600">{entry.name}</span> -
                <span className="text-gray-700">{entry.loc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Available IR Files */}
        <div>
          <h3 className="text-lg font-medium mb-2 text-gray-800">IR Files</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.keys(kernel.irFiles).map((irType) => (
              <div
                key={irType}
                className="bg-gray-50 rounded p-4 border border-gray-200 hover:bg-blue-50 hover:border-blue-200 cursor-pointer transition-colors"
                onClick={() => onViewIR(irType)}
              >
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    {/* SVG icon representing a document/file */}
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-6 w-6 text-blue-600"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                  </div>
                  <div className="ml-4">
                    <h3 className="text-lg font-medium text-gray-800">
                      {irType}
                    </h3>
                    <p className="text-sm text-gray-600 mt-1">
                      View full IR code
                    </p>
                  </div>
                  <div className="ml-auto">
                    {/* Right arrow icon indicating clickable action */}
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 text-gray-400"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default KernelOverview;
