import { useState, useContext, useEffect, useRef } from "react";
import { ThemeContext } from "../ThemeContext";
import useStore from "../Store/store";
import axios from "axios";
import toast from "react-hot-toast";

const ImageUploader = () => {
  const { theme } = useContext(ThemeContext);

  const [image, setImage] = useState<string | null>(null);
  const [fileObj, setFileObj] = useState<File | null>(null);
  const [error, setError] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const ActiveSession = useStore((state) => state.ActiveSession);

  useEffect(() => {
    if (!ActiveSession) {
      setImage(null);
      return;
    }
    setImage(ActiveSession.imageurl || null);
  }, [ActiveSession]);

  // ----------------------------- Handle File -----------------------------
  const handleFiles = (file: File) => {
    setError("");

    const allowedTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!allowedTypes.includes(file.type)) {
      setError("Only JPG, JPEG, PNG files are allowed.");
      return;
    }

    setFileObj(file);

    const reader = new FileReader();
    reader.onload = () => setImage(reader.result as string);
    reader.readAsDataURL(file);
  };

  // ----------------------------- Upload to Cloudinary -----------------------------
  const uploadToCloudinary = async () => {
    if (!fileObj) return;

    setUploading(true);

    const formData = new FormData();
    formData.append("file", fileObj);
    formData.append("upload_preset", "preset");
    formData.append("cloud_name", "dzczys4gk");

    try {
      const res = await fetch(
        `https://api.cloudinary.com/v1_1/dzczys4gk/image/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await res.json();

      const sessionID: string = ActiveSession ? ActiveSession._id : "";

      await axios.patch(
        `${import.meta.env.VITE_GLOBAL_BASE_URL}/update_session/${sessionID}`,
        { imageurl: data.secure_url }
      );
      toast.success("Image uploaded successfully!");
      setUploading(false);
    } catch (err) {
      console.error("Upload failed:", err);
      setError("Upload failed. Try again.");
      setUploading(false);
    }
  };

  // ----------------------------- Drag Events -----------------------------
  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFiles(file);
  };

  return (
    <div className="w-full flex flex-col items-center space-y-4 ">
      {/* MAIN IMAGE PREVIEW (always visible if image exists) */}
      <div
        className={`w-fit h-fit max-w-3xl h-56 rounded-2xl border flex items-center justify-center overflow-hidden
          ${
            theme === "dark"
              ? "bg-[#1a212d] border-[#2a3242]"
              : "bg-gray-100 border-gray-300"
          }
        `}
      >
        {image ? (
          <img
            src={image}
            alt="preview"
            className="w-fit h-fit object-contain rounded-xl"
          />
        ) : (
          <p className="opacity-50">No image selected — Upload below</p>
        )}
      </div>

      {/* ----------------------------- 
          UPLOAD AREA (HIDE WHEN IMAGE EXISTS) 
          ----------------------------- */}
      {!image && (
        <div
          className={`w-full max-w-3xl h-40 cursor-pointer rounded-2xl border-2 flex flex-col 
            items-center justify-center transition-all select-none
            ${isDragging ? "border-blue-500 bg-blue-500/10 scale-[1.02]" : ""}
            ${
              theme === "dark"
                ? "bg-[#1a212d]/40 border-[#2a3242] text-gray-300"
                : "bg-white border-gray-300 text-gray-700"
            }
          `}
          onDrop={onDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onClick={() => fileInputRef.current?.click()}
        >
          <img
            src="https://cdn-icons-png.flaticon.com/512/685/685655.png"
            alt="upload"
            className="w-10 opacity-60 mb-2"
          />
          <span
            className={`text-sm font-semibold underline ${
              theme === "dark" ? "text-blue-400" : "text-blue-600"
            }`}
          >
            Click to upload
          </span>
          <span className="text-xs opacity-70">or drag and drop</span>
        </div>
      )}

      {/* Hidden input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFiles(file);
        }}
      />

      {/* Error */}
      {error && <p className="text-red-500 text-sm">{error}</p>}

      {/* UPLOAD BUTTON (visible only when preview exists AND file selected) */}
      {fileObj && (
        <button
          onClick={uploadToCloudinary}
          disabled={uploading}
          className={`px-6 py-2 rounded-lg text-white font-medium transition
            ${uploading ? "opacity-50 cursor-not-allowed" : ""}
            ${theme === "dark" ? "bg-[#238636]" : "bg-blue-600"}
          `}
        >
          {uploading ? "Uploading..." : "Upload"}
        </button>
      )}
    </div>
  );
};

export default ImageUploader;
