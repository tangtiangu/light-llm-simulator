import * as Vue from "https://unpkg.com/vue@3/dist/vue.esm-browser.prod.js";
import { loadModule } from "https://unpkg.com/vue3-sfc-loader/dist/vue3-sfc-loader.esm.js";

const options = {
  moduleCache: {
    vue: Vue,
  },
  async getFile(url) {
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Could not load ${url}`);
    }
    return await response.text();
  },
  addStyle(textContent) {
    const style = Object.assign(document.createElement("style"), {
      textContent,
    });
    document.head.appendChild(style);
  },
};

const app = Vue.createApp({
  components: {
    TabManager: Vue.defineAsyncComponent(() =>
      loadModule("/static/components/TabManager.vue", options),
    ),
  },
  template: "<TabManager />",
});

app.mount("#app");
