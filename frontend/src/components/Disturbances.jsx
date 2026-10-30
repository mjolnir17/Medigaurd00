import { useState } from "react";
import { pct, pts, tone, nice } from "../utils/format";

const Delta = ({ d }) => <span className={`chip ${tone(d)}`}>{pts(d)}</span>;

export default function Disturbances({ cats, baseline }) {
  const keys = Object.keys(cats);
  const [k, setK] = useState(keys[0]);
  const cat = cats[k];
  return (
    <section className="panel">
      <div>
        <h2>Performance under disturbance</h2>
        <p className="muted">
          Each row is the same model on the same images with one disturbance applied. Changes are measured against the clean results above.
        </p>
      </div>
      <div className="tabs">
        {keys.map((key) => (
          <button key={key} className={key === k ? "on" : ""} onClick={() => setK(key)}>
            {cats[key].display}
          </button>
        ))}
      </div>
      <div className="legend">Bar = accuracy. Orange line = clean accuracy ({pct(baseline.accuracy)}).</div>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Condition</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th>
              <th>Accuracy change</th><th>Specificity change</th>
            </tr>
          </thead>
          <tbody>
            {cat.conditions.map((c) => (
              <tr key={c.label}>
                <td>{nice(c.label)}<small>{c.description}</small></td>
                <td>
                  <div className="bar">
                    <div className="fill" style={{ width: pct(c.metrics.accuracy) }} />
                    <div className="mark" style={{ left: pct(baseline.accuracy) }} />
                  </div>
                  <span className="num">{pct(c.metrics.accuracy)}</span>
                </td>
                <td className="num">{pct(c.metrics.sensitivity)}</td>
                <td className="num">{pct(c.metrics.specificity)}</td>
                <td><Delta d={c.delta.accuracy} /></td>
                <td><Delta d={c.delta.specificity} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
