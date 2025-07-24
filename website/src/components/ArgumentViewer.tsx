/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 */

import React, { useState } from 'react';

// Renders the value distribution (e.g., "16 (2 times, in launches: 1-2)")
const DistributionCell: React.FC<{ data: any }> = ({ data }) => {
    if (!data) return null;
    if (data.diff_type === 'summary') {
        return <span className="text-gray-500 italic">{data.summary_text}</span>;
    }
    if (data.diff_type === 'distribution' && data.values) {
        return (
            <ul className="list-none m-0 p-0 space-y-1">
                {data.values.map((item: any, index: number) => {
                    const launchRanges = item.launches
                        .map((r: any) => (r.start === r.end ? `${r.start + 1}` : `${r.start + 1}-${r.end + 1}`))
                        .join(', ');
                    return (
                        <li key={index}>
                            <span className="font-mono bg-gray-100 px-1 rounded">{JSON.stringify(item.value)}</span>
                            <span className="text-gray-500 text-xs ml-2">({item.count} times, in launches: {launchRanges})</span>
                        </li>
                    );
                })}
            </ul>
        );
    }
    return <span className="font-mono">{JSON.stringify(data)}</span>;
};

// Renders a single row in the ArgumentViewer table
const ArgumentRow: React.FC<{
  argName: string;
  argData: any;
  isDiffViewer?: boolean;
}> = ({ argName, argData, isDiffViewer = false }) => {
  // Case 1: This is a complex argument with internal differences
  if (isDiffViewer && argData.diff_type === "argument_diff") {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const { sames, diffs } = argData;
    const hasSames = Object.keys(sames).length > 0;
    const hasDiffs = Object.keys(diffs).length > 0;

        return (
            <div className="bg-gray-50 border-b border-gray-200 last:border-b-0">
                <div 
                    className="flex items-center space-x-4 p-2 cursor-pointer" 
                    onClick={() => setIsCollapsed(!isCollapsed)}
                >
                    <div className="w-48 flex-shrink-0 font-semibold text-gray-800 break-all">{argName}</div>
                    <div className="flex-1 text-gray-500 italic text-sm flex items-center">
                        Complex argument with internal differences
                        {/* Dropdown arrow icon */}
                        <svg className={`w-4 h-4 ml-2 transform transition-transform ${isCollapsed ? '' : 'rotate-90'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path></svg>
                    </div>
                </div>
                {!isCollapsed && (
                    <div className="pb-2 px-4 space-y-3">
                        {hasSames && (
                            <div>
                                <h6 className="text-sm font-semibold text-gray-600 mb-1">Unchanged Properties</h6>
                                <div className="space-y-1 pl-4">
                                    {Object.entries(sames).map(([key, value]) => (
                                        <div key={key} className="flex items-start text-sm">
                                            <span className="w-28 font-mono text-gray-500 flex-shrink-0">{key}:</span>
                                            <span className="font-mono">{JSON.stringify(value)}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                        {hasDiffs && (
                            <div>
                                <h6 className="text-sm font-semibold text-gray-600 mb-1">Differing Properties</h6>
                                <div className="space-y-2 pl-4">
                                    {Object.entries(diffs).map(([key, value]) => (
                                        <div key={key} className="flex items-start text-sm">
                                            <span className="w-28 font-mono text-gray-500 flex-shrink-0">{key}:</span>
                                            <div className="flex-1"><DistributionCell data={value} /></div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
    }

    // Case 2: This is a simple argument (in the "Sames" table)
    return (
        <div className="flex items-start space-x-4 p-2 border-b border-gray-200 last:border-b-0">
            <div className="w-48 flex-shrink-0 font-semibold text-gray-800 break-all">{argName}</div>
            <div className="w-48 flex-shrink-0 text-gray-600">{argData.type}</div>
            <div className="flex-1 font-mono text-sm">
                {typeof argData.value !== 'object' || argData.value === null ? 
                    <span>{String(argData.value)}</span> : 
                    <pre className="text-xs whitespace-pre-wrap break-all">{JSON.stringify(argData, null, 2)}</pre>
                }
            </div>
        </div>
    );
};

// Main container component
const ArgumentViewer: React.FC<{ args: Record<string, any>; isDiffViewer?: boolean; }> = ({ args, isDiffViewer = false }) => {
    if (!args || Object.keys(args).length === 0) {
        return <div className="text-sm text-gray-500 p-2">No arguments to display.</div>;
    }

    // A "complex view" is needed if we are showing diffs and at least one of them is a complex argument_diff
    const isComplexView = isDiffViewer && Object.values(args).some(arg => arg.diff_type === 'argument_diff');

    return (
        <div className="border border-gray-200 rounded-md bg-white">
            {/* Render header only for the simple, non-complex table view */}
            {!isComplexView && (
                <div className="flex items-center space-x-4 p-2 bg-gray-100 font-bold text-gray-800 border-b border-gray-200">
                    <div className="w-48 flex-shrink-0">Argument Name</div>
                    <div className="w-48 flex-shrink-0">Type</div>
                    <div className="flex-1">Value</div>
                </div>
            )}
            
            {/* Rows */}
            <div>
                {Object.entries(args).map(([argName, argData]) => (
                    <ArgumentRow key={argName} argName={argName} argData={argData} isDiffViewer={isDiffViewer} />
                ))}
            </div>
        </div>
    );
};

export default ArgumentViewer;
