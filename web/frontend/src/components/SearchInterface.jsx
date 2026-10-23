import { useState, useRef, useCallback, useEffect } from 'react';
import {
  YOUTUBE_REGEX,
  ALLOWED_IMAGE_TYPES,
  MAX_IMAGE_SIZE,
  RESULT_COUNTS,
  EXAMPLE_QUERIES,
} from '../utils/constants';
import './SearchInterface.css';

const MODES = [
  { id: 'text', label: 'Text' },
  { id: 'image', label: 'Screenshot' },
  { id: 'video', label: 'YouTube link' },
];

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);

const formatFileSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

export default function SearchInterface({ initialQuery = '', onSearch, onCancel, loading }) {
  const [mode, setMode] = useState('text');
  const [query, setQuery] = useState(initialQuery);
  const [videoUrl, setVideoUrl] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [nResults, setNResults] = useState(5);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const trimmedUrl = videoUrl.trim();
  const urlInvalid = trimmedUrl !== '' && !YOUTUBE_REGEX.test(trimmedUrl);

  const selectImage = useCallback((file) => {
    setError('');

    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
      setError("That file isn't a JPEG, PNG or WebP image.");
      return;
    }

    if (file.size > MAX_IMAGE_SIZE) {
      setError(`That image is ${formatFileSize(file.size)}. The limit is 10 MB.`);
      return;
    }

    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  }, []);

  useEffect(() => () => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
  }, [imagePreview]);

  // A screenshot can be pasted straight from the clipboard while the Screenshot tab is open.
  useEffect(() => {
    if (mode !== 'image') return undefined;
    const handlePaste = (e) => {
      const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith('image/'));
      if (file) {
        e.preventDefault();
        selectImage(file);
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [mode, selectImage]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) selectImage(file);
  }, [selectImage]);

  const removeImage = () => {
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const ready = {
    text: query.trim().length >= 2,
    image: imageFile !== null,
    video: trimmedUrl !== '' && !urlInvalid,
  }[mode];
  const canSubmit = ready && !loading;

  const submit = () => {
    if (!canSubmit) return;
    setError('');

    switch (mode) {
      case 'text':
        onSearch({ mode, query: query.trim(), nResults });
        break;
      case 'image':
        onSearch({ mode, file: imageFile, nResults });
        break;
      case 'video':
        onSearch({ mode, url: trimmedUrl, nResults });
        break;
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    submit();
  };

  const handleTextKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      submit();
    }
  };

  const runExample = (example) => {
    setMode('text');
    setQuery(example);
    onSearch({ mode: 'text', query: example, nResults });
  };

  return (
    <div className="search">
      <form className="search-sheet" onSubmit={handleSubmit} aria-label="Find a hadith's source">
        <div className="mode-switch" role="radiogroup" aria-label="What are you checking?">
          {MODES.map((m) => (
            <label key={m.id} className="mode-option">
              <input
                type="radio"
                name="search-mode"
                value={m.id}
                checked={mode === m.id}
                onChange={() => { setMode(m.id); setError(''); }}
              />
              <span>{m.label}</span>
            </label>
          ))}
        </div>

        <div className="search-field">
          {mode === 'text' && (
            <>
              <label htmlFor="text-query-input" className="sr-only">Hadith text</label>
              <textarea
                id="text-query-input"
                className="search-textarea"
                placeholder="Paste the hadith as you received it, in English"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleTextKeyDown}
                maxLength={5000}
                rows={4}
              />
            </>
          )}

          {mode === 'image' && (!imageFile ? (
            <label
              className={`drop-zone ${dragOver ? 'is-over' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
            >
              <input
                ref={fileInputRef}
                type="file"
                className="sr-only"
                accept=".jpg,.jpeg,.png,.webp"
                onChange={(e) => {
                  const file = e.target.files[0];
                  if (file) selectImage(file);
                }}
              />
              <span className="drop-zone-main">
                Drop a screenshot here, paste it, or <span className="drop-zone-link">choose a file</span>
              </span>
              <span className="drop-zone-hint">
                JPEG, PNG or WebP, up to 10 MB. Urdu, Arabic and English text is read and translated.
              </span>
            </label>
          ) : (
            <div className="image-preview">
              <img src={imagePreview} alt="" className="image-preview-thumbnail" />
              <div className="image-preview-info">
                <span className="image-preview-name">{imageFile.name}</span>
                <span className="image-preview-size">{formatFileSize(imageFile.size)}</span>
              </div>
              <button type="button" className="text-button" onClick={removeImage}>
                Remove
              </button>
            </div>
          ))}

          {mode === 'video' && (
            <>
              <label htmlFor="video-url-input" className="sr-only">YouTube link</label>
              <input
                id="video-url-input"
                type="url"
                inputMode="url"
                className="search-input"
                placeholder="https://www.youtube.com/watch?v=…"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                aria-invalid={urlInvalid}
                aria-describedby="video-url-help"
              />
              <p id="video-url-help" className={`field-help ${urlInvalid ? 'is-error' : ''}`}>
                {urlInvalid
                  ? 'Use a youtube.com/watch, youtu.be or YouTube Shorts link.'
                  : 'The speech and any on-screen text are transcribed and translated to English first. Expect 30 to 60 seconds.'}
              </p>
            </>
          )}
        </div>

        {error && (
          <p className="field-help is-error" role="alert">{error}</p>
        )}

        <div className="search-actions">
          <label className="count-field">
            Show
            <select value={nResults} onChange={(e) => setNResults(Number(e.target.value))}>
              {RESULT_COUNTS.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
            passages
          </label>

          {mode === 'text' && (
            <span className="kbd-hint">{IS_MAC ? '⌘' : 'Ctrl'} + Enter</span>
          )}

          {loading && (
            <button type="button" className="button-secondary" onClick={onCancel}>
              Cancel
            </button>
          )}
          <button type="submit" className="button-primary" disabled={!canSubmit}>
            {loading ? 'Searching…' : 'Find the source'}
          </button>
        </div>
      </form>

      <div className="examples">
        <span className="examples-lead">Or try</span>
        {EXAMPLE_QUERIES.map((example) => (
          <button
            key={example}
            type="button"
            className="example"
            onClick={() => runExample(example)}
            disabled={loading}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
