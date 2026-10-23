// Eight-pointed star (khatam): two overlapping squares, the second turned 45°.
export default function StarMark({ size = 22 }) {
  return (
    <svg
      className="star-mark"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round">
        <rect x="5" y="5" width="14" height="14" />
        <rect x="5" y="5" width="14" height="14" transform="rotate(45 12 12)" />
      </g>
      <circle cx="12" cy="12" r="2.2" fill="currentColor" />
    </svg>
  );
}
