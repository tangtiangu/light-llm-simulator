# Design Document: Direct Calculation Mode for LLM Simulator

## Summary

Add a new "direct calculation mode" to the LLM simulator that computes throughput and latency for specified batchsize configurations without TPOT constraints, while preserving the existing constraint-based search mode.

## Background

### Current Behavior

The simulator currently uses TPOT (Time Per Output Token) as a constraint input:
- **Input**: Hardware type, die count range, kv_len, TPOT targets
- **Process**: Binary search for maximum batchsize satisfying TPOT and memory constraints
- **Output**: CSV files with performance metrics (throughput, e2e_time)

### Problem Statement

Users need to explore performance characteristics without TPOT constraints:
- Want to see throughput/latency for specific batchsize values
- Need batchsize as an input parameter rather than a searched output
- Want to understand trade-offs across different batchsize configurations

## Design Goals

1. Add batchsize (`attn_bs`) as input parameter (list type, like `kv_len`)
2. Remove TPOT constraint when not specified
3. Directly compute performance metrics for given configurations
4. Preserve existing TPOT-constraint mode for backward compatibility
5. Update Webapp UI and API to support new mode

## Architecture

### Dual-Mode System

The simulator will support two modes:

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Constraint Mode** | `--tpot` specified | Binary search for max batchsize satisfying TPOT + memory constraints |
| **Direct Calculation Mode** | `--tpot` not specified | Iterate over specified `attn_bs` list, compute performance directly |

### Mode Detection Logic

```python
if args.tpot and len(args.tpot) > 0:
    # Constraint Mode: use binary search with min_attn_bs/max_attn_bs bounds
    run_search_with_constraint(args)
else:
    # Direct Calculation Mode: iterate over attn_bs list
    if not args.attn_bs:
        raise ValueError("attn_bs must be specified when tpot is not provided")
    run_search_direct(args)
```

## Component Changes

### 1. CLI Parameters (`src/cli/main.py`)

**New Parameter:**
```python
parser.add_argument('--attn_bs', nargs='+', type=int, default=None,
                    help='Attention batch size list for direct calculation mode')
```

**Modified Parameter:**
```python
parser.add_argument('--tpot', nargs='+', type=int, default=None,  # Changed from [20,50,70,100,150]
                    help='TPOT targets for constraint mode. If not specified, uses direct calculation mode')
```

**Retained Parameters (unchanged):**
```python
--min_attn_bs, --max_attn_bs  # Binary search bounds for constraint mode
--min_die, --max_die, --die_step  # Die count range search
--kv_len  # KV length list
--micro_batch_num  # Micro batch numbers list
```

### 2. Configuration (`conf/config.py`)

**New Attribute:**
```python
self.attn_bs = attn_bs  # List of batchsize values for direct calculation
```

**Conditional Initialization:**
```python
if tpot is not None and len(tpot) > 0:
    self.tpot = tpot
    self.attn_bs = None  # Not used in constraint mode
else:
    self.tpot = None
    if attn_bs is None:
        raise ValueError("attn_bs required when tpot not specified")
    self.attn_bs = attn_bs
```

### 3. Search Logic (`src/search/afd.py`)

**New Method: `search_direct()`**

```python
def search_direct(self):
    """Direct calculation mode: iterate over attn_bs list."""
    for ffn_die in range(min_ffn_die, max_ffn_die, ffn_die_step):
        routed_expert_per_die = Config.calc_routed_expert_per_die(...)
        for attn_die in range(attn_die_start, attn_die_end, attn_die_step):
            # Node alignment check (unchanged)
            ...
            for attn_bs in self.config.attn_bs:
                result = self._evaluate_config_direct(attn_bs, attn_die, ffn_die, routed_expert_per_die)
                if result is None:
                    continue  # Only memory constraint fails
                # Calculate throughput
                throughput = attn_bs * self.config.micro_batch_num * attn_die / total_die / result['e2e_time'] / MS_2_SEC
                # Store result
                ...
```

**New Method: `_evaluate_config_direct()`**

```python
def _evaluate_config_direct(self, attn_bs, attn_die, ffn_die, routed_expert_per_die):
    """Evaluate configuration without TPOT constraint."""
    # Memory computation (unchanged)
    kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = ...
    
    # Memory constraints only
    attn_used_memory = kv_size * self.config.micro_batch_num + attn_static_memory
    if attn_used_memory > attn_memory_threshold or ffn_static_memory > ffn_memory_threshold:
        return None  # Only memory fails
    
    # Compute timing (unchanged)
    ...
    
    # NO TPOT constraint check - removed
    
    return {
        'attn_time': attn_time,
        'moe_time': moe_time,
        'e2e_time': e2e_time,
        ...  # All other fields
    }
```

**Dispatcher Method:**

```python
def search(self):
    if self.config.tpot is not None:
        self.search_with_constraint()  # Original binary search
    else:
        self.search_direct()  # New direct calculation
```

### 4. DeepEP Search (`src/search/deepep.py`)

Similar changes to AfdSearch:
- Add `search_direct()` method
- Add `_evaluate_config_direct()` without TPOT constraint
- Add dispatcher logic in `search_bs()` and `search_bs_heterogeneous()`

### 5. Output File Naming

**Constraint Mode (unchanged):**
```
Device-Model-tpot{tpot}-kv_len{kv_len}.csv
```

**Direct Calculation Mode (new):**
```
Device-Model-bs{attn_bs}-kv_len{kv_len}.csv
```

Examples:
- `ASCENDA3_Pod-DEEPSEEK_V3-bs128-kv_len4096.csv`
- `ASCENDDAVID120_ASCEND910B2-DEEPSEEK_V3-bs64-kv_len4096.csv` (heterogeneous)

### 6. Webapp Backend (`webapp/backend/main.py`)

**RunRequest Model:**
```python
class RunRequest(BaseModel):
    serving_mode: str = "AFD"
    model_type: str = "deepseek-ai/DeepSeek-V3"
    device_type: str = "Ascend_A3Pod"
    ...
    tpot: Optional[List[int]] = None  # Changed: default None, optional
    attn_bs: Optional[List[int]] = None  # New: batchsize list
    kv_len: Optional[List[int]] = [4096]
    ...
```

**Argument Sanitization:**
```python
def _sanitize_args(req: RunRequest) -> List[str]:
    args = [...]
    if req.tpot:
        args += ["--tpot"] + [str(x) for x in req.tpot]
    if req.attn_bs:
        args += ["--attn_bs"] + [str(x) for x in req.attn_bs]
    ...
```

**fetch_csv_results Endpoint:**
```python
@api.get("/fetch_csv_results")
def fetch_csv_results(
    device_type: str,
    model_type: str,
    attn_bs: Optional[int] = None,  # New: replaces tpot for direct mode
    tpot: Optional[int] = None,     # Retained: for constraint mode
    kv_len: int,
    serving_mode: str = "AFD",
    ...
):
    # Determine mode and construct filename
    if tpot is not None:
        file_name = f"{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}.csv"
    elif attn_bs is not None:
        file_name = f"{device_type}-{model_type}-bs{attn_bs}-kv_len{kv_len}.csv"
    else:
        raise HTTPException(status_code=400, detail="Either tpot or attn_bs must be specified")
    ...
```

**results Endpoint:**
```python
@api.get("/results")
def list_results(
    model_type: str,
    device_type: str,
    total_die: int,
    attn_bs: Optional[int] = None,  # New
    tpot: Optional[int] = None,     # Retained
    kv_len: int,
    ...
):
    # Construct image paths based on mode
    if tpot is not None:
        throughput_candidates = [
            f"/data/images/throughput/.../{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}.png"
        ]
    elif attn_bs is not None:
        throughput_candidates = [
            f"/data/images/throughput/.../{device_type}-{model_type}-bs{attn_bs}-kv_len{kv_len}.png"
        ]
    ...
```

### 7. Frontend UI (`webapp/frontend/`)

**app.js - DEFAULT_CSV_SELECTION:**
```javascript
const DEFAULT_CSV_SELECTION = {
  servingMode: "AFD",
  deploymentMode: "Heterogeneous",
  deviceType: "ASCENDDAVID120",
  deviceType2: "ASCEND910B2",
  modelType: "DEEPSEEK_V3",
  attnBs: 128,        // New: replaces tpot as default
  kvLen: 4096,
  microBatchNum: 3,
  totalDie: 128,
};
```

**RunForm.vue:**
- Remove TPOT input (or make optional)
- Add Attention Batch Size input:
```vue
<div class="field">
  <label>Attention Batch Size (comma separated)</label>
  <input v-model="attnBsInput" placeholder="64,128,256" />
</div>
```
- Validation: TPOT or attn_bs must be provided (at least one)

**ResultsTab/VisualizationsTab:**
- Update filter parameters: tpot → attnBs
- CSV selection uses batchsize instead of tpot

## Data Flow

### Constraint Mode Flow (unchanged)

```
CLI args (tpot specified)
  → Config (tpot set, attn_bs None)
  → AfdSearch.search_with_constraint()
  → Binary search for max batchsize
  → CSV output (tpot in filename)
```

### Direct Calculation Mode Flow (new)

```
CLI args (tpot not specified, attn_bs list)
  → Config (tpot None, attn_bs list)
  → AfdSearch.search_direct()
  → Iterate over attn_bs values
  → Direct performance computation
  → CSV output (batchsize in filename)
```

## Error Handling

1. **Missing batchsize in direct mode:**
   ```python
   if tpot is None and attn_bs is None:
       raise ValueError("attn_bs must be specified when tpot is not provided")
   ```

2. **Memory constraint violation:**
   - Skip configuration, continue to next
   - Log warning with details

3. **Invalid batchsize values:**
   - Filter out non-positive values
   - Validate list format

## Testing Plan

1. **Unit Tests:**
   - Test mode detection logic
   - Test `_evaluate_config_direct()` without TPOT constraint
   - Test file naming for both modes

2. **Integration Tests:**
   - Run simulator with direct mode: `python src/cli/main.py --attn_bs 64 128 --kv_len 4096`
   - Verify CSV output format
   - Test Webapp API endpoints

3. **Validation:**
   - Compare throughput/latency results with manual calculations
   - Verify memory constraint enforcement
   - Check backward compatibility with constraint mode

## Backward Compatibility

- **Constraint mode unchanged**: All existing scripts/workflows continue to work
- **New mode opt-in**: Requires explicit `--attn_bs` without `--tpot`
- **API compatibility**: Both tpot and attn_bs parameters supported

## Migration Notes

For users wanting to use direct calculation mode:

**Before:**
```bash
python src/cli/main.py --tpot 50 --kv_len 4096
```

**After:**
```bash
python src/cli/main.py --attn_bs 64 128 256 --kv_len 4096
```

## Questions Resolved

| Question | Decision |
|----------|----------|
| batchsize parameter format | List type (comma-separated), like kv_len |
| Die count range search | Retained (min_die, max_die, die_step) |
| Output filename | Use batchsize in filename for direct mode: `bs{attn_bs}` |
| CSV results API | Modified to support attn_bs parameter |
| TPOT mode handling | Retained as optional constraint mode |
| Parameter naming | Keep min_attn_bs/max_attn_bs for constraint mode, add attn_bs for direct mode |

## Implementation Priority

1. CLI parameters and mode detection
2. Config class changes
3. AfdSearch direct calculation method
4. DeepEP direct calculation method
5. Output file naming
6. Webapp API changes
7. Frontend UI changes
8. Testing and validation

---

*Design approved by user: 2026-04-24*