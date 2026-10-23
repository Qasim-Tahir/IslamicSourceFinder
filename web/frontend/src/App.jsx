import { useState, useCallback, useEffect, useRef } from 'react';
import Masthead from './components/Masthead';
import SearchInterface from './components/SearchInterface';
import ProgressTracker from './components/ProgressTracker';
import ResultsFeed from './components/ResultsFeed';
import Library from './components/Library';
import Footer from './components/Footer';
import { searchText, searchImage, searchVideo, getStatus } from './utils/api';

// Text searches are kept in the address bar (?q=…) so a check can be shared as a link
const INITIAL_QUERY = new URLSearchParams(window.location.search).get('q')?.trim() ?? '';

function setQueryParam(query) {
  const params = new URLSearchParams(window.location.search);
  if (query) params.set('q', query);
  else params.delete('q');
  const search = params.toString();
  window.history.replaceState(null, '', `${window.location.pathname}${search ? `?${search}` : ''}`);
}

function describeError(err) {
  // fetch rejects with a TypeError when the server can't be reached at all
  if (err instanceof TypeError) {
    return "Can't reach the search server. Start the backend on port 8000, then search again.";
  }
  return err.message || 'The search failed. Try again.';
}

export default function App() {
  const [status, setStatus] = useState(null);

  // Search state
  const [loading, setLoading] = useState(false);
  const [searchMode, setSearchMode] = useState(null);
  const [results, setResults] = useState(null);
  const [searched, setSearched] = useState(null); // what was compared: { mode, text, original, thumb }
  const [error, setError] = useState('');

  // Video progress state
  const [videoStep, setVideoStep] = useState(null);
  const [videoTranscription, setVideoTranscription] = useState('');

  const abortRef = useRef(null);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() => setStatus({ status: 'offline' }));
  }, []);

  // Free the screenshot thumbnail once a new search replaces it
  useEffect(() => () => {
    if (searched?.thumb) URL.revokeObjectURL(searched.thumb);
  }, [searched]);

  const handleSearch = useCallback(async ({ mode, query, file, url, nResults }) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;

    setLoading(true);
    setSearchMode(mode);
    setResults(null);
    setSearched(null);
    setError('');
    setVideoStep(null);
    setVideoTranscription('');

    try {
      switch (mode) {
        case 'text': {
          setQueryParam(query);
          const response = await searchText(query, nResults, signal);
          setSearched({ mode, text: query });
          setResults(response.results);
          break;
        }

        case 'image': {
          setQueryParam('');
          const response = await searchImage(file, nResults, signal);
          setSearched({
            mode,
            text: response.extracted_query,
            original: response.extracted_urdu,
            thumb: URL.createObjectURL(file),
          });
          setResults(response.results);
          break;
        }

        case 'video': {
          setQueryParam('');
          setVideoStep(1);
          const response = await searchVideo(url, nResults, {
            onProgress: setVideoStep,
            onTranscription: setVideoTranscription,
            signal,
          });
          setSearched({ mode, text: response.transcription, url });
          setResults(response.results);
          break;
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') setError(describeError(err));
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
        setVideoStep(null);
      }
    }
  }, []);

  // Run a search opened from a shared link. Deferred so StrictMode's double mount runs it once.
  useEffect(() => {
    if (INITIAL_QUERY.length < 2) return undefined;
    const timer = setTimeout(() => handleSearch({ mode: 'text', query: INITIAL_QUERY, nResults: 5 }), 0);
    return () => clearTimeout(timer);
  }, [handleSearch]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
    setVideoStep(null);
  }, []);

  return (
    <>
      <Masthead status={status}>
        <SearchInterface
          initialQuery={INITIAL_QUERY}
          onSearch={handleSearch}
          onCancel={handleCancel}
          loading={loading}
        />
      </Masthead>

      <main className="page">
        {videoStep && (
          <ProgressTracker currentStep={videoStep} transcription={videoTranscription} />
        )}

        {error && !loading && (
          <div className="page-notice" role="alert">
            <p className="page-notice-title">Search didn't finish</p>
            <p>{error}</p>
          </div>
        )}

        <ResultsFeed
          results={results}
          loading={loading && searchMode !== 'video'}
          searched={searched}
          totalDocuments={status?.total_documents}
        />

        {!loading && !results && <Library status={status} />}
      </main>

      <Footer />
    </>
  );
}
