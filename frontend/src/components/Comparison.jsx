import { pct } from "../utils/format";
import "../styles/insights.css";

const num = (v) => v.toFixed(1);

function buildRows(models) {
  const cats = Object.keys(models[0].categories);
  return [
    { group: "On clean images", items: [
      ["Accuracy", (m) => m.baseline.accuracy, pct],
      ["Sensitivity", (m) => m.baseline.sensitivity, pct],
      ["Specificity", (m) => m.baseline.specificity, pct],
      ["F1 score", (m) => m.baseline.f1, pct],
    ] },
    { group: "Under disturbance (average accuracy)", items: cats.map((k) => [
      models[0].categories[k].display, (m) => m.categories[k].avg_accuracy, pct,
    ]) },
    { group: "Robustness", items: [
      ["Robustness score", (m) => m.scores.robustness_score, num],
      ["Worst-case score", (m) => m.scores.worst_case_score, num],
    ] },
  ];
}

function insight(models) {
  const pick = (f) => models.reduce((a, b) => (f(b) > f(a) ? b : a));
  const acc = pick((m) => m.baseline.accuracy);
  const rob = pick((m) => m.scores.robustness_score);
  if (acc.key === rob.key) return `${acc.display_name} has both the highest clean accuracy and the highest robustness score.`;
  return `${acc.display_name} has the highest clean accuracy, but ${rob.display_name} holds up better under disturbance. Clean accuracy alone would not reveal this.`;
}

export default function Comparison({ models }) {
  return (
    <section className="panel">
      <div>
        <h2>Model comparison</h2>
        <p className="muted">Every model is tested on the same images with the same disturbances. The best value in each row is highlighted.</p>
      </div>
      <div className="insight">{insight(models)}</div>
      <div className="scroll">
        <table className="compare">
          <thead>
            <tr><th></th>{models.map((m) => <th className="model" key={m.key}>{m.display_name}</th>)}</tr>
          </thead>
          <tbody>
            {buildRows(models).map((g) => [
              <tr className="group" key={g.group}><td colSpan={models.length + 1}>{g.group}</td></tr>,
              ...g.items.map(([label, get, fmt]) => {
                const vals = models.map(get);
                const top = Math.max(...vals);
                return (
                  <tr key={g.group + label}>
                    <td>{label}</td>
                    {vals.map((v, i) => (
                      <td key={i} className={`num ${v === top ? "best" : ""}`}>{fmt(v)}</td>
                    ))}
                  </tr>
                );
              }),
            ])}
          </tbody>
        </table>
      </div>
    </section>
  );
}
