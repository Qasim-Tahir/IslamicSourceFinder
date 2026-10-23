import StarMark from './StarMark';
import { Medallion, CornerPiece } from './Ornaments';
import { collectionsFromSources } from '../utils/sources';
import './Masthead.css';

const DEFAULT_COLLECTIONS = ['Sahih al-Bukhari', 'Sahih Muslim', 'Jami at-Tirmidhi'];
const CORNERS = ['tl', 'tr', 'bl', 'br'];

// "A, B and C", with each name kept on one line
function joinNames(names) {
  return names.map((name, i) => (
    <span key={name}>
      {i > 0 && (i === names.length - 1 ? ' and ' : ', ')}
      <span className="nowrap">{name}</span>
    </span>
  ));
}

function ServerState({ status }) {
  if (!status) return null;
  const healthy = status.status === 'healthy';
  return (
    <span className={`server-state ${healthy ? 'is-up' : 'is-down'}`}>
      <span className="server-state-dot" aria-hidden="true" />
      {healthy ? 'Search server ready' : 'Search server offline'}
    </span>
  );
}

export default function Masthead({ status, children }) {
  const collections = status?.sources?.length
    ? collectionsFromSources(status.sources)
    : DEFAULT_COLLECTIONS;
  const total = status?.total_documents;

  return (
    <header className="masthead">
      <div className="masthead-bar">
        <span className="wordmark">
          <StarMark />
          Islamic Source Finder
        </span>
        <ServerState status={status} />
      </div>

      <div className="cover">
        {CORNERS.map((corner) => (
          <CornerPiece key={corner} className={`cover-corner is-${corner}`} />
        ))}

        <Medallion className="cover-medallion" />

        <h1 className="masthead-title">Where does this hadith come from?</h1>
        <p className="masthead-lede">
          Paste the text you were sent, drop in a screenshot of it, or give a YouTube link.
          It's compared by meaning against {total ? `${total.toLocaleString()} passages from ` : ''}
          {joinNames(collections)}.
        </p>

        {children}
      </div>
    </header>
  );
}
