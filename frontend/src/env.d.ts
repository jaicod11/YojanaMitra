/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the YojanaMitra API, e.g. http://localhost:8000. See .env.example. */
  readonly VITE_API_BASE_URL?: string;
  /** "true" serves the built-in mock instead of calling the API (offline development). */
  readonly VITE_USE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
