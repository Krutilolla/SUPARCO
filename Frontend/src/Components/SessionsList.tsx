import { useEffect, useState, useContext } from "react";
import { ThemeContext } from "../ThemeContext";
import axios from "axios";
import useStore from "../Store/store";

export default function SessionsList() {
  const [activeId, setActiveId] = useState<string | null>(null);
  const { theme } = useContext(ThemeContext);
  const setActiveSession = useStore((state) => state.setActiveSession);
  const [sessions, setSessions] = useState<any[]>([]);
  const [filtered, setFiltered] = useState<any[]>([]);
  const [search, setSearch] = useState("");
  const [debounceValue, setDebounceValue] = useState("");
  const setchats = useStore((state) => state.setchats);
  useEffect(() => {
    const timer = setTimeout(() => setDebounceValue(search), 300);
    return () => clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    const lower = debounceValue.toLowerCase();
    setFiltered(sessions.filter((s) => s.name.toLowerCase().includes(lower)));
  }, [debounceValue, sessions]);

  const getSessions = async () => {
    try {
      const response = await axios.get(
        `${import.meta.env.VITE_GLOBAL_BASE_URL}/get_session`
      );
      setSessions(response.data);
      setFiltered(response.data);
      setActiveId(response.data[0]?._id || null);
      if (response.data[0]) {
        
        setActiveSession(response.data[response.data.size()]);
        setchats(response.data[response.data.size()].chats);
      }
    } catch (err) {
      console.error("Error fetching sessions", err);
    }
  };

  useEffect(() => {
    getSessions();
  }, []);

  return (
    <div
      className={`w-80 h-full rounded-xl p-4 flex flex-col transition-all ${
        theme === "dark"
          ? "bg-[#0f151f] border border-[#1f2838]"
          : "bg-white shadow-sm"
      }`}
    >
      {/* Search Bar */}
      <div
        className={`flex items-center gap-2 rounded-xl px-3 py-2 ${
          theme === "dark"
            ? "bg-[#1b2230] text-gray-300"
            : "bg-gray-100 text-gray-700"
        }`}
      >
        <input
          type="text"
          placeholder="Search Sessions..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className={`flex-1 bg-transparent outline-none text-sm ${
            theme === "dark" ? "text-white" : "text-gray-700"
          }`}
        />
      </div>

      <div className="overflow-y-auto no-scrollbar flex-1 mt-4">
        <Header title="🗂 All Sessions" theme={theme} />

        {filtered.map((s) => (
          <SessionItem
            key={s._id}
            data={{
              id: s._id,
              name: s.name,
              timestamp: new Date(s.createdAt).toLocaleString(),
            }}
            active={activeId === s._id}
            onClick={() => {
              setActiveId(s._id);
              setActiveSession(s);
              setchats(s.chats);
              console.log("Selected Session:", s);
            }}
            theme={theme}
          />
        ))}

        {filtered.length === 0 && (
          <p
            className={`text-center text-sm mt-6 ${
              theme === "dark" ? "text-gray-500" : "text-gray-600"
            }`}
          >
            No sessions found.
          </p>
        )}
      </div>
    </div>
  );
}

function Header({ title, theme }: { title: string; theme: string }) {
  return (
    <div className="flex items-center justify-between px-1 mt-4 mb-2">
      <span
        className={`text-sm font-medium ${
          theme === "dark" ? "text-gray-300" : "text-gray-600"
        }`}
      >
        {title}
      </span>
    </div>
  );
}

function SessionItem({ data, active, onClick, theme }:any) {
  return (
    <div
      onClick={onClick}
      className={`flex items-center justify-between py-3 px-2 rounded-lg cursor-pointer transition ${
        active
          ? theme === "dark"
            ? "bg-[#1f2838]"
            : "bg-[#e1ebf9]"
          : theme === "dark"
          ? "hover:bg-[#16202d]"
          : "hover:bg-gray-100"
      }`}
    >
      <div className="flex flex-col">
        <span
          className={`text-sm font-medium ${
            active
              ? theme === "dark"
                ? "text-white"
                : "text-gray-900"
              : theme === "dark"
              ? "text-gray-300"
              : "text-gray-800"
          }`}
        >
          {data.name}
        </span>
        <span
          className={`text-xs ${
            active
              ? theme === "dark"
                ? "text-gray-400"
                : "text-gray-600"
              : "text-gray-500"
          }`}
        >
          {data.timestamp}
        </span>
      </div>
    </div>
  );
}
