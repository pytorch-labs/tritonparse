import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

// A single frame of a stack trace
const StackTraceFrame: React.FC<{ frame: any }> = ({ frame }) => (
  <div className="font-mono text-xs break-all">
    <span className="text-gray-500">{frame.filename}</span>:
    <span className="font-semibold text-blue-600">{frame.line}</span> in{" "}
    <span className="font-semibold text-green-700">{frame.name}</span>
    {frame.line_code && (
       <div className="pl-6 mt-1 bg-gray-100 rounded">
        <SyntaxHighlighter 
            language="python" 
            style={oneLight} 
            customStyle={{ 
                margin: 0, 
                padding: '0.25em 0.5em', 
                fontSize: '0.75rem',
                background: 'transparent'
             }}
        >
            {frame.line_code}
        </SyntaxHighlighter>
    </div>
    )}
  </div>
);


const StackDiffViewer: React.FC<{ stackDiff: any }> = ({ stackDiff }) => {
  const [isCollapsed, setIsCollapsed] = useState(true);

  if (!stackDiff || stackDiff.diff_type !== 'distribution') {
    return null;
  }

  return (
    <div>
      <h5 
        className="text-md font-semibold mb-2 text-gray-700 cursor-pointer flex items-center"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        Stack Traces
        <svg
          className={`w-4 h-4 ml-2 transform transition-transform ${isCollapsed ? '' : 'rotate-90'}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
        </svg>
      </h5>
      {!isCollapsed && (
         <div className="space-y-2">
          {stackDiff.values.map((item: any, index: number) => {
            const launchRanges = item.launches
              .map((r: any) => (r.start === r.end ? `${r.start + 1}` : `${r.start + 1}-${r.end + 1}`))
              .join(", ");
            
            return (
              <div key={index} className="bg-white p-2 rounded border border-gray-200">
                <p className="text-xs font-semibold text-gray-600 mb-1">
                  Variant seen {item.count} times (in launches: {launchRanges})
                </p>
                <div className="space-y-1 bg-gray-50 p-1 rounded">
                   {Array.isArray(item.value) ? item.value.map((frame: any, frameIndex: number) => (
                    <StackTraceFrame key={frameIndex} frame={frame} />
                  )) : <p>Invalid stack format</p>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default StackDiffViewer; 