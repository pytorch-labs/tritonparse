import React from 'react';

interface ArgumentProps {
  argName: string;
  argData: any;
}

const Argument: React.FC<ArgumentProps> = ({ argName, argData }) => {
  const renderValue = () => {
    if (argData.type === 'tensor') {
      return (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {Object.entries(argData)
            .filter(([key]) => key !== 'type') // Don't repeat the type
            .map(([key, value], index, arr) => (
              <span key={key} className="inline-flex items-center">
                <span className="font-semibold text-gray-600">{key}:</span>
                <span className="ml-1 text-gray-800">{JSON.stringify(value)}</span>
                {index < arr.length - 1 && <span className="ml-2 text-gray-400">,</span>}
              </span>
            ))}
        </div>
      );
    } else if (argData.type === 'int' || typeof argData.value !== 'object') {
      return <span>{argData.value}</span>;
    } else {
      return <pre className="text-xs whitespace-pre-wrap break-all">{JSON.stringify(argData, null, 2)}</pre>;
    }
  };

  return (
    <div className="p-2 border-b border-gray-200">
      <div className="flex items-start space-x-4">
        <div className="w-48 flex-shrink-0 font-semibold text-gray-800 break-all">{argName}</div>
        <div className="w-32 flex-shrink-0 text-gray-600">{argData.type}</div>
        <div className="flex-1 font-mono text-sm">{renderValue()}</div>
      </div>
    </div>
  );
};

interface ArgumentViewerProps {
  args: Record<string, any>;
}

const ArgumentViewer: React.FC<ArgumentViewerProps> = ({ args }) => {
  if (!args || Object.keys(args).length === 0) {
    return <p className="text-sm text-gray-500">No arguments to display.</p>;
  }

  return (
    <div className="bg-white rounded border border-gray-200">
      <div className="p-2 bg-gray-50 border-b border-gray-200 flex items-center space-x-4 text-xs font-bold text-gray-500 uppercase">
        <div className="w-48 flex-shrink-0">Argument Name</div>
        <div className="w-32 flex-shrink-0">Type</div>
        <div className="flex-1">Value / Details</div>
      </div>
      {Object.entries(args).map(([argName, argData]) => (
        <Argument key={argName} argName={argName} argData={argData} />
      ))}
    </div>
  );
};

export default ArgumentViewer;
