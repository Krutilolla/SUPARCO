import { useContext, useState, useRef, useEffect } from "react";
import { ThemeContext } from "../ThemeContext";
import useStore from "../Store/store";
import axios from "axios";
import useTypedMessage from "./useTypedMessage";

/* ------------------ Message Interfaces ------------------ */

interface Message {
  id: number;
  text: string;
  sender: "user" | "bot";
  url?: string; // 👈 ADDED
}

interface ChatApiResponse {
  status: string;
  user_message: string;
  response: string;
  image_url?: string; // 👈 ADDED
}

/* ------------------ Typing Indicator ------------------ */

const TypingBubble = ({ theme }: { theme: string }) => (
  <div
    className={`
      max-w-[70%] p-3 rounded-2xl mt-2 space-y-2 animate-pulse
      ${theme === "dark" ? "bg-[#161b22]" : "bg-gray-200"}
    `}
  >
    <div className={`h-3 w-3/4 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-300"}`} />
    <div className={`h-3 w-1/2 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-300"}`} />
    <div className={`h-3 w-1/3 rounded-md ${theme === "dark" ? "bg-gray-700" : "bg-gray-300"} opacity-80`} />
  </div>
);

/* ------------------ User Message ------------------ */

const UserMessage = ({ text, theme }: { text: string; theme: string }) => (
  <div
    className={`
      max-w-[70%] px-4 py-2 rounded-2xl text-sm shadow-sm ml-auto
      ${theme === "dark" ? "bg-blue-600 text-white" : "bg-blue-100 text-gray-900"}
    `}
  >
    {text}
  </div>
);

/* ------------------ Bot Message ------------------ */

const BotMessage = ({
  text,
  theme,
  image_url,
  onImageClick,
}: {
  text: string;
  theme: string;
  image_url?: string;
  onImageClick: (url: string) => void;
}) => {
  const animated = useTypedMessage(text, 20);

  return (
    <div
      className={`
        max-w-[70%] px-4 py-2 rounded-2xl text-sm shadow-sm mr-auto space-y-2
        ${theme === "dark" ? "bg-[#161b22] text-white" : "bg-gray-100 text-gray-900"}
      `}
    >
      {animated}

      {/* Render Image if Exists */}
      {image_url && (
        <img
          src={image_url}
          className="w-full rounded-xl mt-2 cursor-pointer hover:opacity-80 transition"
          onClick={() => onImageClick(image_url)}
        />
      )}
    </div>
  );
};

/* ------------------ Full Screen Image Viewer ------------------ */

const ImagePreview = () => {
  const previewImage = useStore((s) => s.previewImage);
  const closePreview = useStore((s) => s.closePreview);

  if (!previewImage) return null;

  return (
    <div
      className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-center justify-center z-[9999]"
      onClick={closePreview}
    >
      <img
        src={previewImage}
        className="max-w-[90%] max-height-[90%] rounded-xl shadow-xl"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
};

/* ------------------ Main Chat Component ------------------ */

const taskList = ["Generate Caption", "Generate grounding", "Visual Question Answering"];

const ChatWindow = () => {
  const { theme } = useContext(ThemeContext);
  const ActiveSession = useStore((state) => state.ActiveSession);

  const setPreviewImage = useStore((s) => s.setPreviewImage);

  const [showMenu, setShowMenu] = useState(false);
  const [task, setTask] = useState("Generate Caption");
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [botTyping, setBotTyping] = useState(false);

  const inputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const chats = useStore((state) => state.chats);
  // const setchats = useStore((state) => state.setchats);

  /* ------------------ Auto Scroll ------------------ */

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages, botTyping]);

  /* ------------------ Load Session Chats ------------------ */

  useEffect(() => {
    if (chats) {
      const formatted = chats.map((c: any) => ({
        id: Date.now() + Math.random(),
        text: c.message,
        sender: c.sender,
        image_url: c.url || undefined,
      }));
      setMessages(formatted);
    }
  }, [chats]);

  /* ------------------ Send Message ------------------ */

  const sendMessage = async () => {
    const trimmed = inputValue.trim();
    if (!trimmed || isSending) return;

    const userMsg: Message = {
      id: Date.now(),
      text: trimmed,
      sender: "user",
    };

    setMessages((prev) => [...prev, userMsg]);

    setInputValue("");
    setIsSending(true);
    setBotTyping(true);

    try {
      const url = `${import.meta.env.VITE_GLOBAL_BASE_URL}/chat/${ActiveSession._id}`;

      const res = await axios.post(url, {
        message: trimmed,
        task: task,
      });

      const data: ChatApiResponse = res.data;

      const botMsg: Message = {
        id: Date.now() + 1,
        text: data.response,
        sender: "bot",
        url: data.image_url,
      };

      setTimeout(() => {
        setMessages((prev) => [...prev, botMsg]);
        setBotTyping(false);
      }, 400);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSending(false);
    }
  };

  /* ------------------ Outside Click to Close Task Menu ------------------ */

  const menuRef = useRef<HTMLDivElement | null>(null);
  const menuButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!showMenu) return;
      const target = e.target as Node;
      if (
        menuRef.current &&
        !menuRef.current.contains(target) &&
        menuButtonRef.current &&
        !menuButtonRef.current.contains(target)
      ) {
        setShowMenu(false);
      }
    };

    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showMenu]);

  /* ------------------ UI ------------------ */

  return (
    <>
      <div
        className={`
          flex flex-col h-full w-full rounded-2xl p-4 relative overflow-hidden transition-all
          ${
            theme === "dark"
              ? "bg-[#0f151f] border border-[#1f2838]"
              : "bg-white border border-gray-200"
          }
        `}
      >
        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 no-scrollbar pr-2">
          {messages.map((msg) =>
            msg.sender === "user" ? (
              <UserMessage key={msg.id} text={msg.text} theme={theme} />
            ) : (
              <BotMessage
                key={msg.id}
                text={msg.text}
                theme={theme}
                image_url={msg.url!="" ? msg.url : undefined}
                onImageClick={setPreviewImage}
              />
            )
          )}

          {botTyping && <TypingBubble theme={theme} />}

          <div ref={messagesEndRef} />
        </div>

        {/* Task Menu */}
        {showMenu && (
          <div
            ref={menuRef}
            className={`
              absolute bottom-20 left-4 w-56 p-2 rounded-xl shadow-lg z-10 cursor-pointer
              ${
                theme === "dark"
                  ? "bg-[#161b22] border border-[#1f2838] text-white"
                  : "bg-white border border-gray-200 text-gray-800"
              }
            `}
          >
            {taskList.map((item) => (
              <button
                key={item}
                onClick={() => {
                  setTask(item);
                  setShowMenu(false);
                  inputRef.current?.focus();
                }}
                className={`
                  text-sm w-full px-3 py-2 rounded-lg text-left transition
                  ${
                    item === task
                      ? theme === "dark"
                        ? "bg-blue-600 text-white"
                        : "bg-blue-100 text-blue-700"
                      : theme === "dark"
                      ? "hover:bg-[#1f2838]"
                      : "hover:bg-gray-100"
                  }
                `}
              >
                {item}
              </button>
            ))}
          </div>
        )}

        {/* Input */}
        <div
          className={`
            mt-4 flex items-center gap-2 rounded-full px-4 py-2
            ${
              theme === "dark"
                ? "bg-[#161b22] border border-[#1f2838]"
                : "bg-gray-100 border border-gray-300"
            }
          `}
        >
          {/* Menu Button */}
          <button
            ref={menuButtonRef}
            onClick={() => setShowMenu((prev) => !(prev))}
            className={theme === "dark" ? "text-white text-xl" : "text-black text-xl"}
          >
            ☰
          </button>

          {/* Input field */}
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a message"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            className={`
              w-full bg-transparent outline-none text-sm
              ${
                theme === "dark"
                  ? "text-white placeholder-gray-400"
                  : "text-gray-700 placeholder-gray-500"
              }
            `}
          />

          {/* Send */}
          <button
            onClick={sendMessage}
            disabled={isSending}
            className={`
              p-2 rounded-full text-white cursor-pointer
              ${theme === "dark" ? "bg-blue-600" : "bg-blue-500"}
              ${isSending ? "opacity-50" : ""}
            `}
          >
            ➤
          </button>
        </div>
      </div>

      {/* Fullscreen Image Viewer */}
      <ImagePreview />
    </>
  );
};

export default ChatWindow;
