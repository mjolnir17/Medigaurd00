import { pct } from "../utils/format";

export default function Baseline({ b, meta }) {
  const tiles = [
    ["Accuracy", b.accuracy], ["Sensitivity", b.sensitivity], ["Specificity", b.specificity],
    ["Precision", b.precision], ["F1 score", b.f1], ["Average confidence", b.mean_confidence],
  ];
  const cells = [
    ["Healthy, correctly cleared", b.tn], ["Healthy, false alarm", b.fp],
    ["Diseased, missed", b.fn], ["Diseased, correctly detected", b.tp],
  ];
  return (
    <section className="panel">
      <div>
        <h2>Performance on clean images</h2>
        <p className="muted">
          {meta.n_images} test X-rays with no disturbance applied. This is the reference for every comparison below.
        </p>
      </div>
      <div className="tiles">
        {tiles.map(([l, v]) => (
          <div className="tile" key={l}><div className="v">{pct(v)}</div><div className="l">{l}</div></div>
        ))}
      </div>
      <div className="matrix">
        {cells.map(([l, v]) => (
          <div className="cell" key={l}><div className="v">{v}</div><div className="l">{l}</div></div>
        ))}
      </div>
    </section>
  );
}
