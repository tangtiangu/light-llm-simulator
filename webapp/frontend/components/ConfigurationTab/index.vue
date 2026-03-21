<template>
  <div class="configuration-tab">
    <ModelConfig />
    <HardwareConfig />
  </div>
</template>

<script>
import ModelConfig from './ModelConfig.vue';
import HardwareConfig from './HardwareConfig.vue';

export default {
  components: { ModelConfig, HardwareConfig }
};
</script>

<style scoped>
.configuration-tab {
  max-width: 700px;
}
</style>
