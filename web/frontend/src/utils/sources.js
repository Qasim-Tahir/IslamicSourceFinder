/**
 * Helpers that turn raw search results into readable citations and passages.
 */

import { RELEVANCE_THRESHOLDS } from './constants';

const COLLECTION_NAMES = {
  'bukhari.pdf': 'Sahih al-Bukhari',
  'muslim.pdf': 'Sahih Muslim',
};

const TIRMIDHI_FILE = /^Jami-at-Tirmidhi-Vol-(\d+)-(\d+)-(\d+)\.pdf$/i;

function toNumber(value) {
  return /^\d+$/.test(String(value)) ? Number(value) : null;
}

export function collectionName(pdfName) {
  if (TIRMIDHI_FILE.test(pdfName)) return 'Jami at-Tirmidhi';
  return COLLECTION_NAMES[pdfName] ?? pdfName.replace(/\.pdf$/i, '');
}

const CANONICAL_ORDER = ['Sahih al-Bukhari', 'Sahih Muslim', 'Jami at-Tirmidhi'];

/** Unique collection names from the /api/status source list (one PDF per Tirmidhi volume). */
export function collectionsFromSources(sources = []) {
  const rank = (name) => {
    const i = CANONICAL_ORDER.indexOf(name);
    return i === -1 ? CANONICAL_ORDER.length : i;
  };
  return [...new Set(sources.map(collectionName))].sort((a, b) => rank(a) - rank(b));
}

/**
 * Group the /api/status passage counts by collection, one entry per volume:
 * [{ collection, total, volumes: [{ volume, count }] }] in canonical order.
 */
export function libraryGroups(counts = {}) {
  const groups = new Map();
  for (const [pdfName, count] of Object.entries(counts)) {
    const collection = collectionName(pdfName);
    const volume = pdfName.match(TIRMIDHI_FILE)?.[1];
    if (!groups.has(collection)) groups.set(collection, { collection, total: 0, volumes: [] });
    const group = groups.get(collection);
    group.total += count;
    group.volumes.push({ volume: volume ? Number(volume) : null, count });
  }
  const order = collectionsFromSources(Object.keys(counts));
  return order.map((name) => {
    const group = groups.get(name);
    group.volumes.sort((a, b) => (a.volume ?? 0) - (b.volume ?? 0));
    return group;
  });
}

/**
 * Citation for a result. Bukhari and Muslim use the book/number scheme of the
 * indexed English translations. Tirmidhi has no book numbers, and the parsed
 * hadith number is only kept when it falls inside that volume's range.
 */
export function describeSource({ pdf_name, book_number, hadees_number }) {
  const tirmidhi = pdf_name.match(TIRMIDHI_FILE);
  const number = toNumber(hadees_number);

  if (tirmidhi) {
    const [, volume, first, last] = tirmidhi.map(Number);
    const inRange = number !== null && number >= first && number <= last;
    return {
      collection: 'Jami at-Tirmidhi',
      reference: inRange ? `vol.\u00a0${volume}, no.\u00a0${number}` : `vol.\u00a0${volume}, nos.\u00a0${first}–${last}`,
    };
  }

  const book = toNumber(book_number);
  return {
    collection: collectionName(pdf_name),
    reference: [book !== null && `book\u00a0${book}`, number !== null && `no.\u00a0${number}`]
      .filter(Boolean)
      .join(', '),
  };
}

// Running page headers, e.g. "60 / 1800SAHIH MUSLIM BOOK 1: THE BOOK OF FAITH (KITAB AL-IMAN)"
// and "SAHIH BUKHARI VOLUME 9 > BOOK 93: ONENESS, UNIQUENESS OF ALLAH (TAWHEED)".
// Long book titles wrap, so an all-caps line ending in ")" right after the header goes too.
const PAGE_HEADER = /^.*(?:\d+\s*\/\s*\d+\s*SAHIH MUSLIM|BUKHARI VOLUME \d+ > BOOK \d+:).*$(?:\n[^a-z\n]*\)[ \t]*$)?/gm;

/** Undo PDF extraction artefacts: the citation header, page headers, hard line breaks, split words, OCR dash runs. */
export function cleanPassage(text) {
  return text
    .replace(/^Book [^\n]*\(Relevance: [-\d.]+\)\s*/, '')
    .replace(PAGE_HEADER, '')
    .replace(/\t+/g, ' ')
    .replace(/(\p{L}) -\s*\n\s*(\p{L})/gu, '$1$2')
    .replace(/(?:\s[-–•.](?=\s)){2,}/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/\s*Volume \d+,?\s*$/, '')
    .trim();
}

export function strengthOf(score) {
  if (score >= RELEVANCE_THRESHOLDS.HIGH) return { id: 'close', label: 'Close wording', level: 3 };
  if (score >= RELEVANCE_THRESHOLDS.MEDIUM) return { id: 'related', label: 'Same subject', level: 2 };
  return { id: 'distant', label: 'Distant', level: 1 };
}

// Words too common in these translations to be worth marking.
const STOPWORDS = new Set(
  ('that this with from have were when your they them their there what which will would about into than then '
    + 'said says upon also only been being does each other some such very whom whose while where shall unto '
    + 'allah messenger prophet apostle narrated reported peace blessings him his her one who the and for not')
    .split(' '),
);

function stemKey(word) {
  return word.replace(/(?:ing|ed|es|s|ly)$/, '').slice(0, 5);
}

/** Stems of the meaningful words in the searched text. */
export function buildTerms(query = '') {
  const terms = new Set();
  for (const word of query.toLowerCase().match(/\p{L}[\p{L}']*/gu) ?? []) {
    if (word.length < 4 || STOPWORDS.has(word)) continue;
    const key = stemKey(word);
    if (key.length >= 3) terms.add(key);
  }
  return terms;
}

/** Split text into [{ text, match }] pieces, marking words that share a stem with the search. */
export function markShared(text, terms) {
  if (!terms.size) return [{ text, match: false }];
  return text
    .split(/(\p{L}[\p{L}']*)/u)
    .filter(Boolean)
    .map((piece) => ({
      text: piece,
      match: piece.length >= 4 && terms.has(stemKey(piece.toLowerCase())),
    }));
}

export const hasArabicScript = (text = '') => /[؀-ۿ]/.test(text);
