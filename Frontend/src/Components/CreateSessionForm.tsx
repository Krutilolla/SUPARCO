import { useState, useContext } from "react";
import { ThemeContext } from "../ThemeContext";

export default function SessionNamePopup({
  onSubmit,
  onClose,              // <-- NEW
}: {
  onSubmit: (name: string) => void;
  onClose: () => void;  // <-- NEW
}) {
  const { theme } = useContext(ThemeContext);
  const [name, setName] = useState("");

  return (
    <div
      className="
        fixed inset-0 flex items-center justify-center
        backdrop-blur-md bg-black/10 z-50
      "
      onClick={onClose}     // <-- CLICKING OUTSIDE closes popup
    >
      <div
        className={`
          p-6 rounded-xl shadow-xl w-[350px]
          transition
          ${theme === "dark" ? "bg-[#0d1117] border border-[#263040]" : "bg-white"}
        `}
        onClick={(e) => e.stopPropagation()}  // <-- CLICK INSIDE should NOT close
      >
        <h3
          className={`text-lg font-semibold mb-4 
          ${theme === "dark" ? "text-white" : "text-gray-800"}`}
        >
          Create New Session
        </h3>

        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter session name..."
          className={`
            w-full px-3 py-2 rounded-lg mb-4 outline-none 
            ${theme === "dark" ? "bg-[#161b22] text-white" : "bg-gray-100 text-gray-700"}
          `}
        />

        <button
          onClick={() => name.trim() && onSubmit(name)}
          className={`
            w-full py-2 rounded-lg transition font-medium
            ${theme === "dark" ? "bg-[#238636] text-white" : "bg-blue-600 text-white"}
          `}
        >
          Create
        </button>
      </div>
    </div>
  );
}
