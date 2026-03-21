const API_BASE = "/api";

async function fetchJson(url, options = {}) {
  try {
    const response = await fetch(`${API_BASE}${url}`, {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(
        `API error: ${response.status} ${response.statusText}${detail ? ` - ${detail}` : ""}`,
      );
    }

    return await response.json();
  } catch (error) {
    console.error("API call failed:", error);
    throw error;
  }
}

export function useApi() {
  return {
    startRun(payload) {
      return fetchJson("/run", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },

    getStatus(runId) {
      return fetchJson(`/status/${runId}`);
    },

    getLogs(runId) {
      return fetchJson(`/logs/${runId}`);
    },

    getModelConfig(modelType) {
      return fetchJson(`/model_config?${new URLSearchParams({ model_type: modelType })}`);
    },

    getHardwareConfig(deviceType) {
      return fetchJson(`/hardware_config?${new URLSearchParams({ device_type: deviceType })}`);
    },

    getResults(params) {
      return fetchJson(`/results?${new URLSearchParams(params)}`);
    },

    fetchCsvResults(params) {
      return fetchJson(`/fetch_csv_results?${new URLSearchParams(params)}`);
    },

    getConstants() {
      return fetchJson("/constants");
    },
  };
}
