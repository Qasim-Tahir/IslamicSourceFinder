import { useEffect, useMemo, useRef, useState } from 'react';
import ResultCard from './ResultCard';
import {
  buildTerms,
  cleanPassage,
  describeSource,
  hasArabicScript,
  markShared,
  strengthOf,
} from '../utils/sources';
import './ResultsFeed.css';

const SEARCHED_LABELS = {
  text: 'What you searched',
  image: 'Read from your screenshot',
  video: 'Transcript of the video',
};

function verdictFor(best) {
  const strength = strengthOf(best.relevance_score);
  const { collection, reference } = describeSource(best);

  if (strength.id === 'close') {
    return {
      title: (
        <>
          Close match in <span className="nowrap">{collection}</span>
          {reference && `, ${reference}`}.
        </>
      ),
      body: 'Compare the wording on both pages before you share it.',
    };
  }
  if (strength.id === 'related') {
    return {
      title: 'No close wording, but these passages cover the same subject.',
      body: "The text you have may be a paraphrase, a mix of several narrations, or from a collection that isn't indexed.",
    };
  }
  return {
    title: 'Nothing close to this in the indexed collections.',
    body: 'Only Sahih al-Bukhari, Sahih Muslim and Jami at-Tirmidhi are searched, so it may come from another book. It may also be misattributed.',
  };
}

// Left page: the user's side of the comparison
function SearchedPage({ searched, termsFromMatch }) {
  const [expanded, setExpanded] = useState(false);
  const { thumb } = searched;
  const long = (searched.text?.length ?? 0) > 700;
  const showOriginal = searched.original
    && searched.original.trim() !== searched.text?.trim();
  const pieces = useMemo(
    () => markShared(searched.text ?? '', termsFromMatch),
    [searched.text, termsFromMatch],
  );

  return (
    <div className="book-page is-searched">
      <h3 className="page-label">{SEARCHED_LABELS[searched.mode]}</h3>

      {thumb && <img className="searched-thumb" src={thumb} alt="Your screenshot" />}

      {showOriginal && (
        hasArabicScript(searched.original)
          ? <p className="searched-original" lang="ur" dir="rtl">{searched.original}</p>
          : <p className="searched-original">{searched.original}</p>
      )}

      {searched.text && (
        <>
          {showOriginal && <h4 className="page-sublabel">Searched as</h4>}
          <p className={`searched-text ${long && !expanded ? 'is-clamped' : ''}`}>
            {pieces.map((piece, i) => (piece.match ? <mark key={i}>{piece.text}</mark> : piece.text))}
          </p>
          {long && (
            <button
              type="button"
              className="text-button"
              aria-expanded={expanded}
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? 'Show less' : 'Show all'}
            </button>
          )}
        </>
      )}
    </div>
  );
}

export default function ResultsFeed({ results, loading, searched, totalDocuments }) {
  const headingRef = useRef(null);
  const terms = useMemo(() => buildTerms(searched?.text), [searched]);
  const termsFromMatch = useMemo(
    () => buildTerms(results?.[0] ? cleanPassage(results[0].text) : ''),
    [results],
  );

  useEffect(() => {
    if (!results?.length || !headingRef.current) return;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    headingRef.current.focus({ preventScroll: true });
    headingRef.current.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
  }, [results]);

  if (loading) {
    return (
      <section className="results is-loading" aria-busy="true" aria-live="polite">
        <p className="results-loading-text">
          Searching {totalDocuments ? `${totalDocuments.toLocaleString()} passages` : 'the collections'}…
        </p>
        <div className="open-book is-placeholder" aria-hidden="true">
          <div className="book-page"><span /><span /><span /></div>
          <div className="book-page"><span /><span /><span /><span /></div>
        </div>
      </section>
    );
  }

  if (!results || !searched) return null;

  if (results.length === 0) {
    return (
      <section className="results">
        <div className="verdict">
          <h2 className="verdict-title" ref={headingRef} tabIndex={-1}>No passages came back.</h2>
          <p className="verdict-body">Try a shorter phrase, or the key sentence on its own.</p>
        </div>
      </section>
    );
  }

  const [best, ...others] = results;
  const verdict = verdictFor(best);

  return (
    <section className="results" aria-labelledby="verdict-title">
      <div className="verdict">
        <h2 id="verdict-title" className="verdict-title" ref={headingRef} tabIndex={-1}>
          {verdict.title}
        </h2>
        <p className="verdict-body">{verdict.body}</p>
      </div>

      <div className="open-book">
        <SearchedPage searched={searched} termsFromMatch={termsFromMatch} />
        <div className="book-page is-source">
          <ResultCard result={best} index={0} terms={terms} featured />
        </div>
      </div>
      <p className="book-caption">
        Words that appear on both pages are <mark>marked</mark>.
      </p>

      {others.length > 0 && (
        <div className="others">
          <h3 className="others-title">Other nearby passages</h3>
          <ol className="passages">
            {others.map((result, i) => (
              <ResultCard
                key={`${result.pdf_name}-${result.book_number}-${result.hadees_number}-${i}`}
                result={result}
                index={i + 1}
                terms={terms}
              />
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
