import { useContext, useEffect, useState } from "react";
import SessionsButton from "./Components/Sessions_btn";
import MainSection from "./Components/MainSection";
import Add_btn from "./Components/Add_btn";
import { ThemeContext } from "./ThemeContext";
import "./App.css";
import useStore from "./Store/store";
import {Toaster}  from "react-hot-toast";
function App() {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const { theme } = useContext(ThemeContext);
  const { setShowSessions } = useStore();
  useEffect(() => {
    // Initialize activeIndex and showSessions on mount
    setActiveIndex(0);
    setShowSessions(true);          // ✔ Always open sessions on load
  }, []);
  return (
    <div
      className={`
        h-screen w-screen flex transition-all
        ${theme === "dark" ? "bg-[#0a0f18]" : "bg-[#e1ebf9]"}
      `}
    >
      <Toaster />
      {/* LEFT SIDEBAR */}
      <div
        className={`
          w-24 flex flex-col p-4 gap-4 rounded-xl 
          transition-all m-2
          ${theme === "dark" ? "bg-[#0f151f] border border-[#1f2838]" : "bg-white shadow-md"}
        `}
      >
        <SessionsButton
          active={activeIndex === 0}
          onClick={() => {
            setActiveIndex(0);
            setShowSessions(true);        // ✔ Always open sessions
          }}
        />

        <Add_btn
          active={activeIndex === 1}
          onClick={() => {
            setActiveIndex(1);
            setShowSessions(false);       // ✔ Always hide sessions
          }}
        />
      </div>

      {/* MAIN SECTION */}
      <div className="flex-1 p-2">
        <MainSection />
      </div>
    </div>
  );
}

export default App;