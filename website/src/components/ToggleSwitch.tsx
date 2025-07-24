import React from "react";

interface ToggleSwitchProps {
  isChecked: boolean;
  onChange: (isChecked: boolean) => void;
  label?: string;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({
  isChecked,
  onChange,
  label,
}) => {
  const handleToggle = () => {
    onChange(!isChecked);
  };

  return (
    <div className="flex items-center">
      {label && <span className="mr-2 text-sm font-medium">{label}</span>}
      <label className="relative inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          checked={isChecked}
          onChange={handleToggle}
          className="sr-only peer"
        />
        <div
          className="
            w-11 h-6 bg-gray-200 rounded-full 
            peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 
            peer-checked:bg-blue-600
            after:content-[''] after:absolute after:top-[2px] after:left-[2px] 
            after:bg-white after:border-gray-300 after:border after:rounded-full 
            after:h-5 after:w-5 after:transition-all
            peer-checked:after:translate-x-full peer-checked:after:border-white
          "
        ></div>
      </label>
    </div>
  );
};

export default ToggleSwitch;
