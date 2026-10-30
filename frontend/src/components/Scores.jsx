import { ratingTone } from "../utils/format";

export default function Scores({ scores }) {
  return (
    <section className="scores">
      <div className="score">
        <div className="label">Robustness score (average)</div>
        <div className="big">{scores.robustness_score}</div>
        <div className={ratingTone(scores.rating)}>{scores.rating}</div>
      </div>
      <div className="score">
        <div className="label">Worst-case score</div>
        <div className="big">{scores.worst_case_score}</div>
        <div className={ratingTone(scores.worst_case_rating)}>{scores.worst_case_rating}</div>
      </div>
    </section>
  );
}
