import { ref } from "https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js";

const STORAGE_KEY_TAB = "llm-sim-current-tab";
const STORAGE_KEY_HISTORY = "llm-sim-run-history";
const STORAGE_KEY_CSV_SELECTION = "llm-sim-csv-selection";

function parseStoredJson(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch (error) {
    console.warn(`Failed to parse localStorage for ${key}:`, error);
    return fallback;
  }
}

const currentTab = ref(localStorage.getItem(STORAGE_KEY_TAB) || "run");
const runHistory = ref(parseStoredJson(STORAGE_KEY_HISTORY, []));
const csvSelection = ref({
  servingMode: "AFD",
  deviceType: "ASCENDA3_Pod",
  modelType: "DEEPSEEK_V3",
  tpot: 50,
  kvLen: 4096,
  microBatchNum: 3,
  totalDie: 128,
  ...parseStoredJson(STORAGE_KEY_CSV_SELECTION, {}),
});
const tabs = ["run", "config", "results", "visualizations"];

export function useStore() {
  const setTab = (tab) => {
    currentTab.value = tab;
    localStorage.setItem(STORAGE_KEY_TAB, tab);
  };

  const addRun = (runId, params) => {
    runHistory.value.unshift({
      id: runId,
      params,
      timestamp: Date.now(),
    });

    if (runHistory.value.length > 20) {
      runHistory.value = runHistory.value.slice(0, 20);
    }

    localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(runHistory.value));
  };

  const clearHistory = () => {
    runHistory.value = [];
    localStorage.removeItem(STORAGE_KEY_HISTORY);
  };

  const setCsvSelection = (selection) => {
    csvSelection.value = {
      ...csvSelection.value,
      ...selection,
    };
    localStorage.setItem(STORAGE_KEY_CSV_SELECTION, JSON.stringify(csvSelection.value));
  };

  const getCsvSelection = () => csvSelection.value;

  return {
    currentTab,
    tabs,
    setTab,
    runHistory,
    addRun,
    clearHistory,
    csvSelection,
    setCsvSelection,
    getCsvSelection,
  };
}
