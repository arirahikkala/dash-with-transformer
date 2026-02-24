let container: HTMLElement | null = null;

export function showErrorToast(message: string): void {
  if (!container) {
    container = document.createElement("div");
    Object.assign(container.style, {
      position: "fixed",
      bottom: "1rem",
      right: "1rem",
      display: "flex",
      flexDirection: "column",
      gap: "0.5rem",
      zIndex: "9999",
    });
    document.body.appendChild(container);
  }
  const toast = document.createElement("div");
  Object.assign(toast.style, {
    background: "#d32f2f",
    color: "#fff",
    padding: "0.6rem 1rem",
    borderRadius: "4px",
    fontFamily: "monospace",
    fontSize: "0.85rem",
    maxWidth: "28rem",
    wordBreak: "break-word",
    boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
    opacity: "1",
    transition: "opacity 0.3s",
  });
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 5000);
}
