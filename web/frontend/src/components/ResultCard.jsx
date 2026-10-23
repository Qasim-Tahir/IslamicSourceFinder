import { useState, useCallback, useMemo } from 'react';
import { cleanPassage, describeSource, markShared, strengthOf } from '../utils/sources';
import './ResultCard.css';

const LONG_PASSAGE = 700;

export default function ResultCard({ result, index, terms, featured = false }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(null);

  const { collection, reference } = describeSource(result);
  const citation = reference ? `${collection}, ${reference}` : collection;
  const strength = strengthOf(result.relevance_score);
  const passage = useMemo(() => cleanPassage(result.text), [result.text]);
  const pieces = useMemo(() => markShared(passage, terms), [passage, terms]);
  const long = passage.length > (featured ? LONG_PASSAGE * 1.6 : LONG_PASSAGE);
  const Tag = featured ? 'article' : 'li';

  const copy = useCallback((what) => {
    const value = what === 'reference' ? citation : `"${passage}" (${citation})`;
    navigator.clipboard.writeText(value)
      .then(() => setCopied(what))
      .catch(() => setCopied('failed'))
      .finally(() => setTimeout(() => setCopied(null), 2000));
  }, [citation, passage]);

  return (
    <Tag className={`passage is-${strength.id} ${featured ? 'is-featured' : ''}`}>
      <div className="passage-head">
        <span className="passage-rank" aria-hidden="true">{index + 1}</span>
        <h3 className="passage-cite">
          <span className="passage-collection">{collection}</span>
          {reference && <span className="passage-ref">{reference}</span>}
        </h3>
        <span className="strength" title={`Similarity ${result.relevance_score.toFixed(2)}`}>
          <span className="strength-ticks" aria-hidden="true">
            {[1, 2, 3].map((n) => <i key={n} className={n <= strength.level ? 'on' : ''} />)}
          </span>
          <span className="strength-label">{strength.label}</span>
          <span className="strength-score">{result.relevance_score.toFixed(2)}</span>
        </span>
      </div>

      <p className={`passage-text ${long && !expanded ? 'is-clamped' : ''}`}>
        {pieces.map((piece, i) => (piece.match ? <mark key={i}>{piece.text}</mark> : piece.text))}
      </p>

      <div className="passage-actions">
        {long && (
          <button
            type="button"
            className="text-button"
            aria-expanded={expanded}
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Show less' : 'Read the full passage'}
          </button>
        )}
        <button type="button" className="text-button" onClick={() => copy('reference')}>
          {copied === 'reference' ? 'Reference copied' : 'Copy reference'}
        </button>
        <button type="button" className="text-button" onClick={() => copy('text')}>
          {copied === 'text' ? 'Text copied' : 'Copy text with reference'}
        </button>
        <span className="sr-only" aria-live="polite">
          {copied === 'failed' ? 'Copying failed. Select the text and copy it manually.' : ''}
        </span>
      </div>
    </Tag>
  );
}
