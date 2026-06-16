import { useContext } from "react";
import { ThemeContext } from "../ThemeContext";
import SessionsList from "./SessionsList";
import ToggleButton from "./DarkModeToggle";
import useStore from "../Store/store";
import ChatWindow from "./ChatWindow";
import ImageUploader from "./ImageUploader";
import axios from "axios";
import SessionNamePopup from "./CreateSessionForm";
import { toast } from "react-hot-toast";
export interface Message {
  id: number;
  text: string;
  sender: "user" | "bot";
}

const MainSection = () => {
  const { theme } = useContext(ThemeContext);
  const { showSessions } = useStore();
  const ActiveSession = useStore((state) => state.ActiveSession);
  const setShowSessions = useStore((state) => state.setShowSessions);
if (!showSessions) {
  const createNewSession = async (sessionName: string) => {
    try {
      await axios.post(
        `${import.meta.env.VITE_GLOBAL_BASE_URL}/create_session`,
        { name: sessionName }
      );
      toast.success("Session created successfully!");
      window.location.reload();
    } catch (err) {
      console.error("Failed to create session:", err);
    }
  };

  return (
    <SessionNamePopup
      onSubmit={createNewSession}
      onClose={() => setShowSessions(true)}
    />
  );
}



  return (

    <div
      className={`
        h-full w-full rounded-tl-xl p-4 m-2 shadow-md flex flex-col
        transition-all 
        ${theme === "dark" ? "bg-[#0d1117]" : "bg-white bg-opacity-50"}
      `}
      style={{
        border: theme === "dark" ? "1px solid #263040" : "1px solid #dae1f1",
      }}
    >
      {/* ---------- TOP BAR ---------- */}
      <div className="flex items-center w-full mb-4 px-1">
        {/* Left Buttons */}
        {/* <div className="flex gap-3">
          <button
            className={`px-4 py-1 rounded-full border transition ${
              theme === "dark"
                ? "text-white border-[#263040] bg-[#161b22]"
                : "text-gray-700 border-gray-300 bg-white"
            }`}
          >
            ● New Panel
          </button>

          <button
            className={`px-3 py-1 rounded-full border transition ${
              theme === "dark"
                ? "text-white border-[#263040] bg-[#161b22]"
                : "text-gray-700 border-gray-300 bg-white"
            }`}
          >
            ＋
          </button>
        </div> */}

        {/* Title */}
        <h2
          className={`flex-1 text-center font-semibold ${
            theme === "dark" ? "text-white" : "text-gray-800"
          }`}
        >
          {ActiveSession ? ActiveSession.name : "No Session Selected"}
        </h2>

        {/* Dark Mode Toggle */}
        <div className="ml-auto">
          <ToggleButton />
        </div>
      </div>

      {/* ---------- MAIN LAYOUT ---------- */}
      <div className="flex flex-1 overflow-hidden mt-2">

        {/* RENDER EVERYTHING ONLY IF showSessions = true */}
        {showSessions && (
          <>
            {/* LEFT SESSION LIST */}
            <div className="w-80 h-full overflow-y-auto no-scrollbar pr-2">
              <SessionsList />
            </div>

            {/* MAIN WORKSPACE */}
            <div
              className={`
                flex-1 h-full rounded-xl p-4 flex overflow-hidden transition
                ${theme === "dark" ? "bg-[#0f151f]" : "bg-white"}
              `}
              style={{
                border:
                  theme === "dark"
                    ? "1px solid #263040"
                    : "1px solid rgba(0,0,0,0.1)",
              }}
            >
              {/* LEFT SIDE: IMAGE PREVIEW + UPLOADER */}
              <div className="flex-1 h-full mr-4 flex flex-col overflow-y-auto no-scrollbar">
                <h3
                  className={`mb-4 text-sm font-semibold ${
                    theme === "dark" ? "text-white" : "text-gray-700"
                  }`}
                >
                  Image Play Area
                </h3>

                {/* IMAGE AREA */}
                <div
                  className={`
                    w-full h-[430px] rounded-xl transition mb-4
                    ${theme === "dark" ? "bg-[#161b22]" : "bg-gray-100"}
                  `}
                ></div>

                {/* IMAGE UPLOADER */}
                <div className="flex justify-center w-full mb-4">
                  <ImageUploader />
                </div>
              </div>

              {/* RIGHT SIDE: CHAT WINDOW */}
              <div className="w-[450px] h-full">
                <ChatWindow  />
              </div>
            </div>
          </>
        )}

      </div>
    </div>
  );
};

export default MainSection;
