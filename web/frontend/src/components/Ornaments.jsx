// Gilt tooling for the cover: a central medallion (shamsa) and corner pieces.
// Strokes with the "draw" class are traced in once on load (see Masthead.css).

const RING_DOTS = Array.from({ length: 16 }, (_, i) => {
  const angle = (i * Math.PI) / 8;
  return [60 + 52 * Math.cos(angle), 60 + 52 * Math.sin(angle)];
});

export function Medallion({ className }) {
  return (
    <svg className={className} viewBox="0 0 120 120" aria-hidden="true" focusable="false">
      <g fill="none" stroke="currentColor" strokeLinejoin="round">
        <circle className="draw" pathLength="1" cx="60" cy="60" r="57" strokeWidth="1.2" />
        <circle className="draw" pathLength="1" cx="60" cy="60" r="47" strokeWidth="0.8" style={{ '--delay': '150ms' }} />
        <rect className="draw" pathLength="1" x="27" y="27" width="66" height="66" strokeWidth="1.4" style={{ '--delay': '300ms' }} />
        <rect
          className="draw"
          pathLength="1"
          x="27"
          y="27"
          width="66"
          height="66"
          strokeWidth="1.4"
          transform="rotate(45 60 60)"
          style={{ '--delay': '450ms' }}
        />
        <circle className="draw" pathLength="1" cx="60" cy="60" r="17" strokeWidth="1" style={{ '--delay': '600ms' }} />
      </g>
      <g className="ornament-fill" fill="currentColor">
        {RING_DOTS.map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r="1.5" />)}
        <circle cx="60" cy="60" r="4" />
      </g>
    </svg>
  );
}

// Drawn for the top-left corner; the other three are mirrored in CSS.
export function CornerPiece({ className }) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true" focusable="false">
      <g fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round">
        <path className="draw" pathLength="1" d="M42 2 A40 40 0 0 1 2 42" />
        <path className="draw" pathLength="1" d="M30 2 A28 28 0 0 1 2 30" style={{ '--delay': '150ms' }} />
        <rect
          className="draw"
          pathLength="1"
          x="9"
          y="9"
          width="10"
          height="10"
          transform="rotate(45 14 14)"
          style={{ '--delay': '300ms' }}
        />
      </g>
      <circle className="ornament-fill" cx="14" cy="14" r="1.6" fill="currentColor" />
    </svg>
  );
}
