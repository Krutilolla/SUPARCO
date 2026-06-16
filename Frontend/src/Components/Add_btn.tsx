import { useContext, useState } from "react";
import { ThemeContext } from "../ThemeContext";
interface AddBtnProps {
  active: boolean;
  onClick: () => void;
}

export default function Add_btn({ active, onClick }: AddBtnProps) {
  const { theme } = useContext(ThemeContext);
  const [selcted,setSelected]=useState(false);

  const bgColor =
    active
      ? theme === "dark"
        ? "#1a1e28" 
        : "#6c86bfff" 
      : theme === "dark"
      ? "#0f172a" 
      : "#e1ebf9"; 

  // Dynamic text color
  const textColor =
    active
      ? theme === "dark"
        ? "#ffffff" 
        : "#6c86bfff" 
      : theme === "dark"
      ? "#9ca3af" 
      : "#97b6f9ff"; 

  const strokeColor = theme === "dark" ? "#9ca3af" : "#6b6b6b";

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        onClick={()=>{
          onClick();
          setSelected(!selcted);
        }}
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
          width="36"
          height="36"
          viewBox="0 0 36 36"
          fill="none"
          stroke={strokeColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <rect
            x="3"
            y="3"
            width="30"
            height="30"
            rx="15"
            ry="15"
            strokeDasharray="4 4"
          />
          <line x1="18" y1="11" x2="18" y2="25" />
          <line x1="11" y1="18" x2="25" y2="18" />
        </svg>
      </button>

      <p
        className="text-sm text-center transition"
        style={{
          color: textColor,
          fontWeight: active ? 600 : 400
        }}
      >
        Add Sessions
      </p>
    </div>
  );
}
