/**
 * API configuration constants.
 */

// In development, the Vite proxy handles /api forwarding.
// In production, set this to your backend URL.
export const API_BASE_URL = '';

export const ENDPOINTS = {
  SEARCH_TEXT: '/api/search/text',
  SEARCH_IMAGE: '/api/search/image',
  SEARCH_VIDEO: '/api/search/video',
  STATUS: '/api/status',
};

// Cosine similarity cut-offs. Picked from sample queries against the current index:
// verbatim hadith score around 0.6-0.7, same-topic passages 0.45-0.6, unrelated below that.
export const RELEVANCE_THRESHOLDS = {
  HIGH: 0.60,
  MEDIUM: 0.45,
};

export const YOUTUBE_REGEX = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)[\w-]+/;

export const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
export const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];

// The backend accepts 1-20.
export const RESULT_COUNTS = [3, 5, 10, 15, 20];

// Order matches the step numbers sent by /api/search/video.
export const VIDEO_STEPS = [
  'Fetching the audio from YouTube',
  'Uploading it to Gemini',
  'Transcribing and translating',
  'Searching the collections',
];

// The last one is a widely shared saying with no close match in these collections.
export const EXAMPLE_QUERIES = [
  'The strong man is not the one who wrestles, but the one who controls himself when angry',
  'None of you believes until he loves for his brother what he loves for himself',
  'Seek knowledge even if you have to go to China',
];
