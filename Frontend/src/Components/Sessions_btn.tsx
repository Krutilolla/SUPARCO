import { useContext } from "react";
import { ThemeContext } from "../ThemeContext";
interface SessionsButtonProps {
  active: boolean;
  onClick: () => void;
}

export default function SessionsButton({ active, onClick }: SessionsButtonProps) {
  const { theme } = useContext(ThemeContext);
    
 

  // Background color based on theme + active
  const bgColor =
    active
      ? theme === "dark"
        ? "#1a1e28"     // active dark
        : "#6c86bfff"   // active light
      : theme === "dark"
      ? "#0f172a"       // inactive dark
      : "#e1ebf9";      // inactive light

  // Icon color
  const iconColor =
    active
      ? theme === "dark"
        ? "white"       // active dark
        : "white"       // active light
      : theme === "dark"
      ? "#9ca3af"       // inactive dark
      : "black";        // inactive light

  // Text color
  const textColor =
    active
      ? theme === "dark"
        ? "white"
        : "#6c86bfff"
      : theme === "dark"
      ? "#9ca3af"
      : "#97b6f9ff";

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        onClick={onClick}
        className="
          cursor-pointer
          w-10 h-15
          rounded-3xl
          flex items-center justify-center
          shadow-md
          active:scale-95
          transition
          p-2
        "
        style={{ backgroundColor: bgColor }}
      >
        <svg
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke={iconColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7A8.38 8.38 0 0 1 8.8 19L3 21l2-5.8a8.5 8.5 0 1 1 16  -3.7z" />
        </svg>
      </button>

      <p
        className="text-sm text-center transition"
        style={{
          color: textColor,
          fontWeight: active ? 600 : 400,
        }}
      >
        Sessions
      </p>
    </div>
  );
}
