import StarMark from './StarMark';
import { libraryGroups } from '../utils/sources';
import './Library.css';

const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'];
const BINDINGS = {
  'Sahih al-Bukhari': 'is-green',
  'Sahih Muslim': 'is-oxblood',
  'Jami at-Tirmidhi': 'is-indigo',
};

export default function Library({ status }) {
  const groups = libraryGroups(status?.source_counts);
  if (!groups.length) return null;

  return (
    <section className="library" aria-labelledby="library-title">
      <h2 id="library-title" className="library-title">What gets searched</h2>
      <p className="library-lede">
        The English translations of three of the six major hadith collections. Each spine is
        as wide as the number of passages indexed from that book.
      </p>

      <ul className="shelf">
        {groups.map((group) => (
          <li
            key={group.collection}
            className={`shelf-group ${BINDINGS[group.collection] ?? 'is-green'}`}
            style={{ flexGrow: group.total }}
          >
            <div className="shelf-spines" aria-hidden="true">
              {group.volumes.map(({ volume, count }) => (
                <span
                  key={volume ?? 'single'}
                  className={`spine ${volume ? 'is-volume' : ''}`}
                  style={{ flexGrow: count }}
                  title={`${group.collection}${volume ? `, vol. ${volume}` : ''}: ${count.toLocaleString()} passages`}
                >
                  {volume ? (
                    <span className="spine-numeral">{ROMAN[volume - 1] ?? volume}</span>
                  ) : (
                    <StarMark size={18} />
                  )}
                </span>
              ))}
            </div>
            <p className="shelf-label">
              <span className="shelf-name">{group.collection}</span>
              <span className="shelf-count">
                {group.total.toLocaleString()} passages
                {group.volumes.length > 1 && ` in ${group.volumes.length} volumes`}
              </span>
            </p>
          </li>
        ))}
      </ul>
    </section>
  );
}
