import React, { useState } from "react";

interface WelcomeScreenProps {
  loadDefaultData: () => void;
  handleFileSelected: (file: File) => void;
  openUrlInput: () => void;
}

/**
 * Welcome screen component shown when no data is loaded
 */
const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ loadDefaultData, handleFileSelected, openUrlInput }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  // Handle file input change
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Support NDJSON and compressed files only
      const fileName = file.name.toLowerCase();
      const isValidFile = fileName.endsWith(".ndjson") || fileName.endsWith(".gz");

      if (isValidFile) {
        handleFileSelected(file);
      }
    }
  };

  // Handle drag and drop events
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Support NDJSON and compressed files only
      const fileName = file.name.toLowerCase();
      const isValidFile = fileName.endsWith(".ndjson") || fileName.endsWith(".gz");

      if (isValidFile) {
        handleFileSelected(file);
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center px-4 py-16 max-w-4xl mx-auto text-center">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Welcome to TritonParse</h2>
      <p className="mb-8 text-gray-600">
        Load a Triton log file to analyze compiled kernels and their IR representations. Supports NDJSON and compressed
        (.gz) files.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-4xl mb-10">
        {/* Default Example Card */}
        <div
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
          onClick={loadDefaultData}
          style={{ cursor: "pointer" }}
        >
          <div className="bg-blue-50 p-3 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
            {/* This SVG icon represents a document or file with a copy/duplicate action.
                It shows one document being copied to another, symbolizing the "Default Example"
                functionality where a sample file is loaded for the user. */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6 text-blue-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Default Example</h3>
          <p className="text-sm text-gray-600">Load the included example Triton log file</p>
        </div>

        {/* Local File Card */}
        <div
          className={`bg-white p-6 rounded-lg shadow-sm border border-gray-200 relative h-52 transition-all duration-200 ${isDragOver ? 'border-green-400 bg-green-200' : ''
            }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="bg-green-50 p-3 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
            {/* This SVG icon represents a document with a plus sign, symbolizing the "Local File"
                functionality where users can add/upload their own file from their device.
                The plus sign indicates the action of adding a new file to the application. */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6 text-green-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Local File</h3>
          <p className="text-sm text-gray-600 mb-4">
            {isDragOver ? 'Drop Triton log file here' : 'Open or drag a Triton log file (NDJSON or .gz)'}
          </p>
          <label htmlFor="welcomeFileInput" className="absolute inset-0 cursor-pointer" aria-label="Open local file" />
          <input
            type="file"
            id="welcomeFileInput"
            accept=".ndjson,.gz,application/x-ndjson,application/gzip"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>

        {/* Remote URL Card */}
        <div
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
          onClick={openUrlInput}
          style={{ cursor: "pointer" }}
        >
          <div className="bg-purple-50 p-3 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
            {/* This SVG icon represents a link or chain, symbolizing the "Remote URL"
                functionality where users can load data from an external web address.
                The chain links indicate connectivity to external resources on the internet. */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6 text-purple-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Remote URL</h3>
          <p className="text-sm text-gray-600">Load a Triton log file from a URL</p>
        </div>
      </div>

      <div className="text-sm text-gray-500 max-w-2xl">
        <h4 className="font-medium mb-2">About TritonParse</h4>
        <p>
          TritonParse helps you analyze Triton GPU kernels by visualizing the compilation process across different IR
          stages.
        </p>
      </div>
    </div>
  );
};

export default WelcomeScreen;
