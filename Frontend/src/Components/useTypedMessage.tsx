import { useState, useEffect } from "react";

export default function useTypedMessage(fullText: string, speed = 20) {
  const [typed, setTyped] = useState("");

  useEffect(() => {
    let i = 0;

    const interval = setInterval(() => {
      setTyped(fullText.slice(0, i));
      i++;
      if (i > fullText.length) clearInterval(interval);
    }, speed);

    return () => clearInterval(interval);
  }, [fullText, speed]);

  return typed;
}
