export const API = "http://127.0.0.1:8000";

const get = (path) =>
  fetch(`${API}${path}`).then((r) => {
    if (!r.ok) throw new Error(`Request failed: ${path}`);
    return r.json();
  });

export const fetchModels = () => get("/models");
export const fetchResults = (key) => get(`/results/${key}`);
