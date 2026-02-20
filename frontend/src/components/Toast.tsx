"use client";

import { createContext, useContext, useState, useCallback, useEffect } from "react";

type ToastType = "info" | "success" | "warning" | "error";

interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (message: string, type?: ToastType, duration?: number) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((message: string, type: ToastType = "info", duration: number = 5000) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setToasts((prev) => [...prev, { id, message, type, duration }]);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  );
}

function ToastContainer({ toasts, onRemove }: { toasts: Toast[]; onRemove: (id: string) => void }) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2.5 max-w-sm">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={onRemove} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: string) => void }) {
  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(() => {
        onRemove(toast.id);
      }, toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.id, toast.duration, onRemove]);

  const typeStyles: Record<ToastType, { border: string; icon: string; iconColor: string }> = {
    info: { border: "border-terminal-mid", icon: "i", iconColor: "text-terminal-mid" },
    success: { border: "border-semantic-pass", icon: "OK", iconColor: "text-semantic-pass" },
    warning: { border: "border-semantic-unknown", icon: "!!", iconColor: "text-semantic-unknown" },
    error: { border: "border-semantic-fail", icon: "XX", iconColor: "text-semantic-fail" },
  };

  const style = typeStyles[toast.type];

  return (
    <div
      className={`flex items-start gap-2.5 px-3.5 py-3 bg-terminal-black border border-dashed ${style.border} rounded-sm shadow-lg animate-slide-left`}
    >
      <span className={`font-mono text-[0.7rem] ${style.iconColor} flex-shrink-0`}>{style.icon}</span>
      <p className="font-mono text-[0.72rem] text-terminal-white leading-relaxed flex-1">{toast.message}</p>
      <button
        onClick={() => onRemove(toast.id)}
        className="text-terminal-dim hover:text-terminal-white transition-colors text-sm flex-shrink-0"
        aria-label="Dismiss"
      >
        x
      </button>
    </div>
  );
}
