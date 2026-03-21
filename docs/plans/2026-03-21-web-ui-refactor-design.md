# Web UI Refactor Design

**Date:** 2026-03-21
**Status:** Approved

## Problem Statement

The Light LLM Simulator has an accurate AFD simulation engine but suffers from poor user experience and layout:

1. No real-time progress feedback during simulations
2. Confusing parameter inputs with no documentation
3. Four-panel layout requires horizontal scrolling
4. Poor result visualization and comparison capabilities

## Solution Overview

Refactor the web frontend to Vue.js 3 (via CDN) while keeping the FastAPI backend unchanged. This provides:

- Tab-based navigation with vertical single-column layout
- Real-time simulation progress
- Filterable, sortable results with run comparison
- Interactive visualizations
- Improved parameter UX with tooltips

## Architecture

### Tech Stack
- **Frontend:** Vue 3 (Composition API) via CDN
- **Charts:** Chart.js (already in use)
- **Backend:** FastAPI (unchanged - calls CLI as subprocess)
- **Persistence:** localStorage for tab state, run history

### Component Structure
```
App
├── TabManager (handles tab switching)
├── RunExperimentTab
│   ├── RunForm (parameter inputs with tooltips)
│   └── RunStatus (real-time progress + log streaming)
├── ConfigurationTab
│   ├── ModelConfig (displays model specs)
│   └── HardwareConfig (displays hardware specs)
├── ResultsTab
│   ├── ResultsFilter (multi-criteria filters)
│   ├── ResultsTable (sortable, paginated, row selection)
│   └── ComparisonSelector (select runs to compare)
└── VisualizationsTab
    ├── ThroughputCharts (interactive scatter/line plots)
    ├── PipelineCharts (Gantt charts with zoom/pan)
    └── CustomComparison (user-selected multi-run comparison)
```

### State Management
- `useAppStore` composable for global state (current tab, run history)
- Reactive refs for component-local state
- `watch()` for config changes (model_type, device_type)

### API Layer
- `api.js` module wrapping all FastAPI calls
- Centralized error handling
- Response caching where appropriate

## Tab Design

### Run Experiment Tab
- Collapsible parameter sections with tooltips
- Real-time progress bar and log streaming (poll `/logs/{run_id}`)
- Phase indicators and estimated time remaining
- Run history in localStorage

### Configuration Tab
- Model specs from `/api/model_config`
- Hardware specs from `/api/hardware_config`
- Formatted display with human-readable units

### Results Tab
- Filter controls: model, device, serving mode, TPOT, KV length, die range
- Sortable data grid with pagination
- Row selection for comparison (checkboxes)
- Side-by-side comparison modal

### Visualizations Tab
- Throughput charts: interactive scatter/line with hover tooltips
- Pipeline charts: Gantt visualization with zoom/pan
- Custom comparison: multi-run side-by-side charts
- Export functionality (PNG/SVG)

## File Structure

```
webapp/
├── backend/
│   └── main.py (unchanged)
└── frontend/
    ├── index.html (Vue app mount point)
    ├── index_old.html (backup of current implementation)
    ├── app.js (Vue app entry)
    ├── components/
    │   ├── TabManager.vue
    │   ├── RunExperimentTab/
    │   ├── ConfigurationTab/
    │   ├── ResultsTab/
    │   └── VisualizationsTab/
    ├── composables/
    │   ├── useApi.js
    │   ├── useStore.js
    │   └── useLocalStorage.js
    └── styles/
        └── main.css
```

## Testing Strategy

### Manual Testing Checklist
- All tabs switch without errors
- Run form submits and returns run_id
- Progress polling updates UI every 3 seconds
- Results filter works with all combinations
- Sortable table columns maintain order
- Comparison view opens with 2+ selected rows
- All charts render with sample data
- Mobile/tablet responsiveness check

### Regression Prevention
- Keep current `index.html` as backup
- Verify all existing API endpoints still work
- Run simulator CLI to confirm backend unchanged

### Cross-Browser Testing
- Chrome (primary)
- Firefox
- Safari (Mac)

## Implementation Notes

- Backend API endpoints remain unchanged
- Use existing `/logs/{run_id}` endpoint for progress streaming
- Use existing `/fetch_csv_results` and `/results` for data
- No build step required - Vue 3 via CDN
- Progressive migration possible (tabs can be implemented incrementally)
