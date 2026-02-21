/**
 * Lightweight logger with level-based filtering.
 *
 * Controlled via the VITE_LOG_LEVEL env var. Defaults to "debug" in
 * development and "warn" in production. Prefix tags ([DEBUG], [INFO],
 * [WARN], [ERROR]) enable easy filtering in browser devtools.
 */

const LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };

function getLogLevel() {
  const env = import.meta.env.VITE_LOG_LEVEL;
  if (env && LEVELS[env] !== undefined) return LEVELS[env];
  return import.meta.env.DEV ? LEVELS.debug : LEVELS.warn;
}

const currentLevel = getLogLevel();

const logger = {
  debug: (...args) => {
    if (currentLevel <= LEVELS.debug) console.debug("[DEBUG]", ...args);
  },
  info: (...args) => {
    if (currentLevel <= LEVELS.info) console.info("[INFO]", ...args);
  },
  warn: (...args) => {
    if (currentLevel <= LEVELS.warn) console.warn("[WARN]", ...args);
  },
  error: (...args) => {
    if (currentLevel <= LEVELS.error) console.error("[ERROR]", ...args);
  },
};

export default logger;
