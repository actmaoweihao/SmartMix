const API_HOST = window.location.hostname || "127.0.0.1";
const API_PROTOCOL = window.location.protocol === "https:" ? "https:" : "http:";

export const API_BASE_URL = `${API_PROTOCOL}//${API_HOST}:8002`;

export function apiUrl(path) {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function fetchJson(path, options = {}) {
  let response;
  try {
    response = await fetch(apiUrl(path), options);
  } catch (error) {
    throw new Error(`无法连接后端 ${API_BASE_URL}。请确认已运行 pnpm backend，或重新运行 pnpm dev。原始错误：${error.message}`);
  }
  if (!response.ok) {
    let message = response.statusText;
    try {
      const error = await response.json();
      message = error.detail || message;
    } catch {
      // Keep the status text.
    }
    throw new Error(message);
  }
  return response.json();
}
