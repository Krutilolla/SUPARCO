import { create } from "zustand";

interface Store {
  showSessions: boolean;
  setShowSessions: (val: boolean) => void;

  ActiveSession: any;
  setActiveSession: (val: any) => void;

  showAdd: boolean;
  setShowAdd: (val: boolean) => void;

  chats: any[] | null;
  setchats: (val: any[]) => void;

  // 👇 ADDED FOR IMAGE PREVIEW FEATURE
  previewImage: string | null;
  setPreviewImage: (url: string) => void;
  closePreview: () => void;
}

const useStore = create<Store>((set) => ({
  showSessions: true,
  setShowSessions: (val) => set({ showSessions: val }),

  ActiveSession: null,
  setActiveSession: (val) => set({ ActiveSession: val }),

  showAdd: false,
  setShowAdd: (val) => set({ showAdd: val }),

  chats: null,
  setchats: (val) => set({ chats: val }),

  // ⭐ FULLSCREEN IMAGE PREVIEW STATE
  previewImage: null,
  setPreviewImage: (url) => set({ previewImage: url }),
  closePreview: () => set({ previewImage: null }),
}));

export default useStore;
