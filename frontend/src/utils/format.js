export const pct = (v) => (v * 100).toFixed(1) + "%";
export const pts = (d) => (d > 0 ? "+" : "") + (d * 100).toFixed(1) + " pts";
export const tone = (d) => (d <= -0.1 ? "bad" : d <= -0.03 ? "mid" : "ok");
export const ratingTone = (r) => ({ Excellent: "ok", Good: "ok", Moderate: "mid" }[r] || "bad");
export const nice = (s) => s.replace(/_/g, " ");
