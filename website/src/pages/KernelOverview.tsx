import React from "react";
import ArgumentViewer from "../components/ArgumentViewer";
import DiffViewer from "../components/DiffViewer";
import { ProcessedKernel } from "../utils/dataLoader";

interface KernelOverviewProps {
  kernels: ProcessedKernel[];
  selectedKernel: number;
  onSelectKernel: (index: number) => void;
  onViewIR: (irType: string) => void;
}

/**
 * Determines if a metadata value is considered "long" and should be displayed at the end
 */
const isLongValue = (value: any): boolean => {
  const formattedString = formatMetadataValue(value);
  return formattedString.length > 50;
};

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
  span = 1,
}) => (
  <div
    className={`flex flex-col ${span > 1 ? `col-span-${span}` : ""} ${
      span === 0 ? "col-span-full" : ""
    }`}
  >
    <span className="text-sm font-medium text-gray-500">{label}</span>
    <span className="font-mono text-sm break-words">{value}</span>
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
              {/* Short fields in responsive grid */}
              <div className="grid grid-cols-[repeat(auto-fit,_minmax(180px,_1fr))] gap-3 mb-4">
                {/* All short metadata fields */}
                {Object.entries(kernel.metadata)
                  .filter(([_key, value]) => !isLongValue(value))
                  .map(([key, value]) => {
                    return (
                      <MetadataItem
                        key={key}
                        label={key
                          .split("_")
                          .map(
                            (word) =>
                              word.charAt(0).toUpperCase() + word.slice(1)
                          )
                          .join(" ")}
                        value={formatMetadataValue(value)}
                      />
                    );
                  })}
              </div>

              {/* Long fields in separate section within same container */}
              {Object.entries(kernel.metadata).filter(([_key, value]) =>
                isLongValue(value)
              ).length > 0 && (
                <div className="space-y-3 border-t border-gray-200 pt-4">
                  {Object.entries(kernel.metadata)
                    .filter(([_key, value]) => isLongValue(value))
                    .map(([key, value]) => (
                      <div key={key} className="w-full">
                        <span className="text-sm font-medium text-gray-500 block mb-1">
                          {key
                            .split("_")
                            .map(
                              (word) =>
                                word.charAt(0).toUpperCase() + word.slice(1)
                            )
                            .join(" ")}
                        </span>
                        <span className="font-mono text-sm block break-all">
                          {formatMetadataValue(value)}
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Launch Analysis Section */}
        {kernel.launchDiff && (
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-3 text-gray-800">
              Launch Analysis
            </h3>
            <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
              <p className="text-sm text-gray-700 mb-4">
                <span className="font-semibold">Total Launches:</span>{" "}
                {kernel.launchDiff.total_launches}
              </p>

              {/* Launch Index Map */}
              {kernel.launchDiff.launch_index_map && (
                <div className="mb-4">
                  <h4 className="text-md font-semibold mb-2 text-gray-800">
                    Launch Locations in Original Trace{" "}
                    <span className="text-sm font-normal text-gray-500">
                      (1-based line numbers)
                    </span>
                  </h4>
                  <div className="font-mono text-sm bg-gray-100 p-2 rounded">
                    {kernel.launchDiff.launch_index_map
                      .map((r: any) =>
                        r.start === r.end
                          ? `${r.start}`
                          : `${r.start}-${r.end}`
                      )
                      .join(", ")}
                  </div>
                </div>
              )}

              {/* Unchanged Fields */}
              {kernel.launchDiff.sames && Object.keys(kernel.launchDiff.sames).length > 0 && (
              <div className="mb-4">
                <h4 className="text-md font-semibold mb-2 text-gray-800">
                  Unchanged Launch Arguments
                </h4>
                <ArgumentViewer args={kernel.launchDiff.sames.extracted_args} />
              </div>
              )}

              {(() => {
                const otherSames = Object.fromEntries(
                  Object.entries(kernel.launchDiff.sames).filter(
                    ([key]) =>
                      key !== "compilation_metadata" &&
                      key !== "extracted_args" &&
                      key !== "event_type"
                  )
                );

                if (Object.keys(otherSames).length > 0) {
                  return (
                    <div className="mb-4">
                      <h4 className="text-md font-semibold mb-2 text-gray-800">
                        Other Unchanged Fields
                      </h4>
                      <div className="grid grid-cols-[repeat(auto-fit,_minmax(180px,_1fr))] gap-3 p-2 bg-white rounded border border-gray-200">
                        {Object.entries(otherSames).map(([key, value]) => (
                          <MetadataItem
                            key={key}
                            label={key
                              .split("_")
                              .map(
                                (word) =>
                                  word.charAt(0).toUpperCase() + word.slice(1)
                              )
                              .join(" ")}
                            value={formatMetadataValue(value)}
                          />
                        ))}
                      </div>
                    </div>
                  );
                }
                return null;
              })()}

              {/* Differing Fields */}
              <div className="mb-4">
                <h4 className="text-md font-semibold mb-2 text-gray-800">
                  Differing Fields
                </h4>
                <DiffViewer diffs={kernel.launchDiff.diffs} />
              </div>
            </div>
          </div>
        )}
        
        {/* Stack Trace */}
        <div className="mb-4">
          <h3 className="text-lg font-medium mb-2 text-gray-800">
            Compilation Stack Trace
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
