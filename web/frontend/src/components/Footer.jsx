import StarMark from './StarMark';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <StarMark size={28} />
        <h2 className="footer-title">How to read the results</h2>
        <div className="footer-columns">
          <p>
            Passages are matched by meaning with sentence embeddings, so a paraphrase or a
            translation can still find its source. A close match tells you where similar wording
            appears. It is not a grading of the hadith. Check the reference in a printed edition,
            or ask someone qualified, before relying on it.
          </p>
          <p>
            Book and number references follow the English translations that were indexed. The
            Tirmidhi volumes were scanned, so some of their passages contain OCR errors. Screenshots
            and videos are read and translated by Gemini before the search runs.
          </p>
        </div>
        <p className="footer-stack">
          Built with ChromaDB, sentence-transformers, Gemini, FastAPI and React.
        </p>
      </div>
    </footer>
  );
}
