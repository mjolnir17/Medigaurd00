import { useEffect, useState } from "react";
import { fetchModels, fetchResults } from "./api";
import Scores from "./components/Scores";
import Baseline from "./components/Baseline";
import Disturbances from "./components/Disturbances";

export default function App() {
  const [models, setModels] = useState([]);
  const [key, setKey] = useState(null);
  const [data, setData] = useState(null);
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

  return (
    <main className="app">
      <header>
        <h1>Mediguard</h1>
        <p className="muted">How reliable is a medical imaging model when image quality is not perfect?</p>
      </header>
      {error && <div className="notice bad">{error}</div>}
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
        </>
      )}
      {!data && !error && key && <div className="notice muted">Loading results...</div>}
    </main>
  );
}
