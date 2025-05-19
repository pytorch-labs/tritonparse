import React from "react";

interface CallstackProps {
  callstack: string[];
}

const Callstack: React.FC<CallstackProps> = ({ callstack }) => {
  return (
    <div className="bg-white rounded-lg p-4 mb-4 shadow border border-gray-200">
      <h2 className="text-xl font-semibold mb-2 text-gray-800">Callstack</h2>
      <div className="bg-gray-50 rounded p-2 overflow-auto max-h-48 border border-gray-200">
        {callstack.map((item, index) => (
          <div
            key={index}
            className="font-mono text-sm py-1 text-gray-700 border-b border-gray-200 last:border-0"
          >
            {index + 1}. {item}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Callstack;
