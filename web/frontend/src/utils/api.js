/**
 * API client for the Islamic Source Finder backend.
 * Handles text search, image search, video search (SSE), and status.
 */

import { API_BASE_URL, ENDPOINTS } from './constants';

async function errorFrom(res, fallback) {
  const body = await res.json().catch(() => ({}));
  const detail = Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail;
  return new Error(detail || `${fallback} (HTTP ${res.status})`);
}

/**
 * Perform a text-based semantic search.
 * @param {string} query - The search query text.
 * @param {number} nResults - Number of results to return.
 * @param {AbortSignal} [signal] - Cancels the request.
 * @returns {Promise<Object>} { results, query, total_results }
 */
export async function searchText(query, nResults = 5, signal) {
  const res = await fetch(`${API_BASE_URL}${ENDPOINTS.SEARCH_TEXT}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, n_results: nResults }),
    signal,
  });

  if (!res.ok) throw await errorFrom(res, 'Search failed');
  return res.json();
}

/**
 * Search by uploaded image (Gemini OCR + translation, then semantic search).
 * @param {File} file - The image file to process.
 * @param {number} nResults - Number of results to return.
 * @param {AbortSignal} [signal] - Cancels the request.
 * @returns {Promise<Object>} { results, extracted_urdu, extracted_query }
 */
export async function searchImage(file, nResults = 5, signal) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('n_results', nResults.toString());

  const res = await fetch(`${API_BASE_URL}${ENDPOINTS.SEARCH_IMAGE}`, {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!res.ok) throw await errorFrom(res, 'Reading the screenshot failed');
  return res.json();
}

/**
 * Search by video URL using Server-Sent Events for progress tracking.
 * @param {string} url - The YouTube URL.
 * @param {number} nResults - Number of results to return.
 * @param {Object} handlers
 * @param {Function} handlers.onProgress - (step: 1-4, message) => void
 * @param {Function} handlers.onTranscription - (text) => void
 * @param {AbortSignal} [handlers.signal] - Cancels the stream.
 * @returns {Promise<Object>} { results, transcription, total_results }
 */
export async function searchVideo(url, nResults = 5, { onProgress, onTranscription, signal } = {}) {
  // fetch instead of EventSource, which only supports GET
  const res = await fetch(`${API_BASE_URL}${ENDPOINTS.SEARCH_VIDEO}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, n_results: nResults }),
    signal,
  });

  if (!res.ok) throw await errorFrom(res, 'Video search failed');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) throw new Error('The server closed the connection before sending results.');

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || ''; // Keep incomplete event in buffer

    for (const block of events) {
      let eventType = '';
      let eventData = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) eventType = line.slice(7);
        else if (line.startsWith('data: ')) eventData = line.slice(6);
      }
      if (!eventType || !eventData) continue;

      let data;
      try {
        data = JSON.parse(eventData);
      } catch (parseErr) {
        console.warn('Failed to parse SSE event:', parseErr);
        continue;
      }

      switch (eventType) {
        case 'progress':
          onProgress?.(Number(data.step), data.message);
          break;
        case 'transcription':
          onTranscription?.(data.text);
          break;
        case 'results':
          reader.cancel();
          return data;
        case 'error':
          reader.cancel();
          throw new Error(data.detail || data.message || 'Video search failed.');
      }
    }
  }
}

/**
 * Fetch system status and collection stats.
 * @returns {Promise<Object>} { status, total_documents, sources, ... }
 */
export async function getStatus() {
  const res = await fetch(`${API_BASE_URL}${ENDPOINTS.STATUS}`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}
