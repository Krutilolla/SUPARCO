import { useContext } from "react";
import { ThemeContext } from "../ThemeContext";

export default function ToggleButton() {
  const { theme, toggleTheme } = useContext(ThemeContext);

  const isDark = theme === "dark";

  return (
    <button
      onClick={toggleTheme}
      className={`
        flex items-center gap-2 px-4 py-1 rounded-full border
        transition-all cursor-pointer select-none active:scale-95
        ${isDark ? "bg-[#0d4ed8] border-[#0d4ed8]" : "bg-white border-gray-300"}
      `}
    >
      {/* Smooth Switch Track */}
      <div
        className={`
          w-12 h-6 rounded-full relative flex items-center
          transition-all duration-300 ease-in-out
          ${isDark ? "bg-blue-600/40" : "bg-gray-300/70"}
        `}
      >
        {/* Sliding circle */}
        <div
          className={`
            w-5 h-5 rounded-full bg-white shadow-md absolute
            transition-all duration-300 ease-in-out
            ${isDark ? "translate-x-6" : "translate-x-1"}
          `}
          style={{
            boxShadow: isDark
              ? "0 0 8px rgba(59,130,246,0.7)"
              : "0 0 4px rgba(0,0,0,0.15)",
          }}
        ></div>
      </div>

      {/* Label */}
      <span
        className={`
          text-sm font-medium transition-colors duration-300
          ${isDark ? "text-white" : "text-gray-700"}
        `}
      >
        {isDark ? "Dark" : "Light"}
      </span>
    </button>
  );
}
