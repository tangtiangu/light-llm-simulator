<template>
  <div class="run-status">
    <h3>Simulation Progress</h3>

    <div v-if="runId" class="status-card">
      <div class="status-header">
        <span class="run-id">Run ID: {{ runId }}</span>
        <button class="btn btn-sm" @click="stopPolling" v-if="!isDone">
          Cancel
        </button>
      </div>

      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
      </div>

      <div class="phase-indicator">{{ currentPhase }}</div>

      <div class="log-container">
        <div v-for="(log, idx) in logs" :key="idx" class="log-line">
          {{ log }}
        </div>
        <div v-if="logs.length === 0" class="log-placeholder">Waiting for logs...</div>
      </div>

      <div v-if="isDone" class="done-message">
        Simulation Complete
      </div>
    </div>

    <div v-else class="no-run">
      No active simulation
    </div>
  </div>
</template>

<script>
import { ref, computed, watch, onUnmounted } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useApi } from '../../composables/useApi.js';
import { useStore } from '../../composables/useStore.js';

export default {
  setup() {
    const api = useApi();
    const { runHistory } = useStore();
    const activeRunId = computed(() => runHistory.value[0] && runHistory.value[0].id ? runHistory.value[0].id : null);

    const runId = ref(null);
    const isDone = ref(false);
    const logs = ref([]);
    const currentPhase = ref('Initializing');
    const progressPercent = ref(0);

    let pollInterval = null;

    const phases = [
      { pattern: /search_attn_bs/i, label: 'Finding optimal attention batch size' },
      { pattern: /searching/i, label: 'Searching die allocations' },
      { pattern: /throughput/i, label: 'Generating visualizations' }
    ];

    const detectPhase = (logText) => {
      for (let i = 0; i < phases.length; i += 1) {
        const phase = phases[i];
        if (phase.pattern.test(logText)) {
          currentPhase.value = phase.label;
          return;
        }
      }
    };

    const stopPolling = () => {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    };

    const pollLogs = async () => {
      if (!runId.value) return;

      try {
        const data = await api.getLogs(runId.value);
        logs.value = (data.log || '').split('\n').filter(Boolean);

        if (logs.value.length > 0) {
          detectPhase(logs.value[logs.value.length - 1]);
        }

        progressPercent.value = Math.min(logs.value.length * 2, 95);
      } catch (err) {
        console.error('Polling failed:', err);
      }
    };

    const pollStatus = async () => {
      if (!runId.value) return;

      try {
        const data = await api.getStatus(runId.value);
        if (data.done) {
          isDone.value = true;
          progressPercent.value = 100;
          currentPhase.value = 'Complete';
          stopPolling();
        }
      } catch (err) {
        console.error('Status polling failed:', err);
      }
    };

    const startPolling = () => {
      stopPolling();
      if (!runId.value) return;

      pollLogs();
      pollStatus();
      pollInterval = setInterval(() => {
        pollLogs();
        pollStatus();
      }, 3000);
    };

    watch(activeRunId, (nextRunId) => {
      stopPolling();
      runId.value = nextRunId;
      logs.value = [];
      isDone.value = false;
      currentPhase.value = nextRunId ? 'Initializing' : 'Idle';
      progressPercent.value = 0;

      if (nextRunId) {
        startPolling();
      }
    }, { immediate: true });

    onUnmounted(() => {
      stopPolling();
    });

    return {
      runId,
      isDone,
      logs,
      currentPhase,
      progressPercent,
      stopPolling
    };
  }
};
</script>

<style scoped>
.run-status {
  max-width: 600px;
}

.status-card {
  background: #f8fafc;
  border-radius: 8px;
  padding: 16px;
  margin-top: 16px;
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.run-id {
  font-weight: 600;
  color: #374151;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 12px;
}

.progress-bar {
  height: 12px;
  background-color: #e5e7eb;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 12px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #10b981);
  transition: width 0.3s ease;
}

.phase-indicator {
  font-size: 13px;
  color: #64748b;
  margin-bottom: 12px;
}

.log-container {
  background: #1f2937;
  color: #f0fdf4;
  border-radius: 6px;
  padding: 12px;
  max-height: 200px;
  overflow-y: auto;
  font-family: 'Monaco', 'Courier New', monospace;
  font-size: 12px;
}

.log-line {
  padding: 4px 0;
  border-bottom: 1px solid #374151;
}

.log-placeholder {
  color: #64748b;
  text-align: center;
  padding: 24px;
}

.done-message {
  text-align: center;
  padding: 16px;
  background-color: #d1fae5;
  color: #065f46;
  border-radius: 6px;
  font-weight: 600;
}

.no-run {
  text-align: center;
  padding: 40px;
  color: #9ca3af;
}
</style>
