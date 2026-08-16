/**
 * Slider UI component
 * @module
 */

import * as React from "react";

export interface SliderProps {
  min?: number;
  max?: number;
  step?: number;
  value?: number[];
  onValueChange?: (value: number[]) => void;
  className?: string;
}

const Slider = React.forwardRef<HTMLDivElement, SliderProps>(
  ({ min = 0, max = 100, step = 1, value = [0], onValueChange, className = "" }, ref) => {
    const [localValue, setLocalValue] = React.useState(value);

    React.useEffect(() => {
      setLocalValue(value);
    }, [value]);

    const handleChange = (index: number, newVal: string) => {
      const numVal = parseFloat(newVal);
      const newValue = [...localValue];
      newValue[index] = numVal;
      setLocalValue(newValue);
      onValueChange?.(newValue);
    };

    return (
      <div ref={ref} className={`flex items-center gap-2 ${className}`}>
        {localValue.map((val, index) => (
          <input
            key={index}
            type="range"
            min={min}
            max={max}
            step={step}
            value={val}
            onChange={(e) => handleChange(index, e.target.value)}
            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        ))}
      </div>
    );
  },
);

Slider.displayName = "Slider";

export { Slider };
