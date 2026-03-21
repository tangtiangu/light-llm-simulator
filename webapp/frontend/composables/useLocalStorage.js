import { ref, watch } from "https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js";

function parseStorage(storage, key, defaultValue) {
  try {
    const stored = storage.getItem(key);
    return stored ? JSON.parse(stored) : defaultValue;
  } catch (error) {
    console.warn(`Failed to parse storage key ${key}:`, error);
    return defaultValue;
  }
}

function createStorageRef(storage, key, defaultValue) {
  const state = ref(parseStorage(storage, key, defaultValue));

  watch(
    state,
    (newValue) => {
      storage.setItem(key, JSON.stringify(newValue));
    },
    { deep: true },
  );

  return state;
}

export function useLocalStorage(key, defaultValue = null) {
  return createStorageRef(localStorage, key, defaultValue);
}

export function useSessionStorage(key, defaultValue = null) {
  return createStorageRef(sessionStorage, key, defaultValue);
}
