<template>
  <div class="run-experiment-tab">
    <RunForm />
    <RunStatus v-if="activeRunId" />
  </div>
</template>

<script>
import { computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js';
import { useStore } from '../../composables/useStore.js';
import RunForm from './RunForm.vue';
import RunStatus from './RunStatus.vue';

export default {
  components: { RunForm, RunStatus },
  setup() {
    const { runHistory } = useStore();
    const activeRunId = computed(() => runHistory.value[0] && runHistory.value[0].id ? runHistory.value[0].id : null);

    return { activeRunId };
  }
};
</script>

<style scoped>
.run-experiment-tab {
  max-width: 600px;
}
</style>
