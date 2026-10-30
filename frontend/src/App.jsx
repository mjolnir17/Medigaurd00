import { useEffect, useState } from "react";
import { fetchModels, fetchResults, fetchCompare } from "./api";
import Scores from "./components/Scores";
import Baseline from "./components/Baseline";
import Disturbances from "./components/Disturbances";
import Risk from "./components/Risk";
import Comparison from "./components/Comparison";
import "./styles/insights.css";

export default function App() {
  const [view, setView] = useState("single");
  const [models, setModels] = useState([]);
  const [key, setKey] = useState(null);
  const [data, setData] = useState(null);
  const [cmp, setCmp] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels()
      .then((m) => { setModels(m); if (m.length) setKey(m[0].key); })
      .catch(() => setError("Cannot reach the API. Start it with: python -m uvicorn api:app --port 8000"));
  }, []);

  useEffect(() => {
    if (!key) return;
    setData(null);
    fetchResults(key).then(setData).catch(() => setError("Could not load results."));
  }, [key]);

  useEffect(() => {
    if (view === "compare" && !cmp) {
      fetchCompare().then((r) => setCmp(r.models)).catch(() => setError("Could not load the comparison."));
    }
  }, [view, cmp]);

  return (
    <main className="app">
      <header>
        <h1>Mediguard</h1>
        <p className="muted">How reliable is a medical imaging model when image quality is not perfect?</p>
      </header>
      {error && <div className="notice bad">{error}</div>}

      <div className="views">
        <button className={view === "single" ? "on" : ""} onClick={() => setView("single")}>Single model</button>
        <button className={view === "compare" ? "on" : ""} onClick={() => setView("compare")}>Compare models</button>
      </div>

      {view === "single" && (
        <>
          <div className="selector">
            {models.map((m) => (
              <button key={m.key} className={m.key === key ? "on" : ""} onClick={() => setKey(m.key)}>
                {m.display_name}
              </button>
            ))}
          </div>
          {data && (
            <>
              <Scores scores={data.scores} />
              <Baseline b={data.baseline} meta={data.meta} />
              <Disturbances cats={data.categories} baseline={data.baseline} />
              <Risk flags={data.risk_flags} recs={data.recommendations} />
            </>
          )}
          {!data && !error && key && <div className="notice muted">Loading results...</div>}
        </>
      )}

      {view === "compare" && (cmp ? <Comparison models={cmp} /> : !error && <div className="notice muted">Loading comparison...</div>)}
    </main>
  );
}
