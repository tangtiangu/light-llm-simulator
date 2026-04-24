# Direct Calculation Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add direct calculation mode to LLM simulator that computes throughput/latency for specified batchsize configurations without TPOT constraints, while preserving existing constraint-based search mode.

**Architecture:** Dual-mode system - Constraint Mode (TPOT specified, binary search) and Direct Calculation Mode (TPOT not specified, iterate over attn_bs list). Changes to CLI, Config, Search classes, Webapp API, and Frontend UI.

**Tech Stack:** Python 3.8+, FastAPI, Vue.js 3

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/cli/main.py` | CLI entry point, argument parsing, mode dispatch | Modify |
| `conf/config.py` | Configuration dataclass with mode-aware initialization | Modify |
| `src/search/afd.py` | AFD search logic, add direct calculation method | Modify |
| `src/search/deepep.py` | DeepEP search logic, add direct calculation method | Modify |
| `webapp/backend/main.py` | FastAPI backend, RunRequest model, API endpoints | Modify |
| `webapp/frontend/app.js` | Frontend state management, CSV selection defaults | Modify |
| `webapp/frontend/components/RunExperimentTab/RunForm.vue` | Run experiment form, add attn_bs input | Modify |
| `webapp/frontend/components/ResultsTab/CsvSelector.vue` | CSV selector, change tpot to attnBs | Modify |
| `webapp/frontend/components/VisualizationsTab/ThroughputCharts.vue` | Charts visualization, change tpot to attnBs | Modify |
| `docs/superpowers/specs/2026-04-24-direct-calculation-mode-design.md` | Design specification | Reference |

---

## Task 1: CLI Parameters and Mode Detection

**Files:**
- Modify: `src/cli/main.py:14-68`

### Step 1.1: Add new --attn_bs argument

In `src/cli/main.py`, add the new argument after line 52 (after `--ffn_tensor_parallel`):

```python
parser.add_argument('--attn_bs', nargs='+', type=int, default=None,
                    help='Attention batch size list for direct calculation mode. Required when tpot is not specified.')
```

### Step 1.2: Modify --tpot default value

Change the default value of `--tpot` from `[20, 50, 70, 100, 150]` to `None`:

```python
# Before:
parser.add_argument('--tpot', nargs='+', type=int, default=[20, 50, 70, 100, 150])

# After:
parser.add_argument('--tpot', nargs='+', type=int, default=None,
                    help='TPOT targets for constraint mode. If not specified, uses direct calculation mode with attn_bs.')
```

### Step 1.3: Add mode validation in run_search function

Modify `run_search` function to validate mode parameters and dispatch accordingly. Replace the entire `run_search` function (lines 70-132):

```python
def run_search(args):
    """
    Run the search to find the best configuration and the best throughput.
    Parameters:
        args: The arguments from the parser.
    """
    # Mode validation
    if args.tpot is None or len(args.tpot) == 0:
        # Direct calculation mode
        if args.attn_bs is None or len(args.attn_bs) == 0:
            raise ValueError("--attn_bs must be specified when --tpot is not provided")
        mode = "direct"
    else:
        # Constraint mode
        if args.attn_bs is not None and len(args.attn_bs) > 0:
            print("Warning: --attn_bs is ignored when --tpot is specified (using constraint mode)")
        mode = "constraint"
    
    if args.serving_mode == "AFD":
        for mbn in args.micro_batch_num:
            for kv_len in args.kv_len:
                if mode == "constraint":
                    for tpot in args.tpot:
                        config = Config(
                            serving_mode=args.serving_mode,
                            model_type=args.model_type,
                            device_type=args.device_type,
                            min_attn_bs=args.min_attn_bs,
                            max_attn_bs=args.max_attn_bs,
                            min_die=args.min_die,
                            max_die=args.max_die,
                            die_step=args.die_step,
                            tpot=tpot,
                            attn_bs=None,
                            kv_len=kv_len,
                            micro_batch_num=mbn,
                            next_n=args.next_n,
                            multi_token_ratio=args.multi_token_ratio,
                            attn_tensor_parallel=args.attn_tensor_parallel,
                            ffn_tensor_parallel=args.ffn_tensor_parallel,
                            deployment_mode=args.deployment_mode,
                            device_type2=args.device_type2,
                            min_die2=args.min_die2,
                            max_die2=args.max_die2,
                            die_step2=args.die_step2
                        )
                        afd_search = AfdSearch(config)
                        afd_search.deployment()
                else:
                    # Direct calculation mode
                    config = Config(
                        serving_mode=args.serving_mode,
                        model_type=args.model_type,
                        device_type=args.device_type,
                        min_attn_bs=args.min_attn_bs,
                        max_attn_bs=args.max_attn_bs,
                        min_die=args.min_die,
                        max_die=args.max_die,
                        die_step=args.die_step,
                        tpot=None,
                        attn_bs=args.attn_bs,
                        kv_len=kv_len,
                        micro_batch_num=mbn,
                        next_n=args.next_n,
                        multi_token_ratio=args.multi_token_ratio,
                        attn_tensor_parallel=args.attn_tensor_parallel,
                        ffn_tensor_parallel=args.ffn_tensor_parallel,
                        deployment_mode=args.deployment_mode,
                        device_type2=args.device_type2,
                        min_die2=args.min_die2,
                        max_die2=args.max_die2,
                        die_step2=args.die_step2
                    )
                    afd_search = AfdSearch(config)
                    afd_search.deployment()
    elif args.serving_mode == "DeepEP":
        for kv_len in args.kv_len:
            if mode == "constraint":
                for tpot in args.tpot:
                    config = Config(
                        serving_mode=args.serving_mode,
                        model_type=args.model_type,
                        device_type=args.device_type,
                        min_attn_bs=args.min_attn_bs,
                        max_attn_bs=args.max_attn_bs,
                        min_die=args.min_die,
                        max_die=args.max_die,
                        die_step=args.die_step,
                        tpot=tpot,
                        attn_bs=None,
                        kv_len=kv_len,
                        micro_batch_num=1,
                        next_n=args.next_n,
                        multi_token_ratio=args.multi_token_ratio,
                        attn_tensor_parallel=args.attn_tensor_parallel,
                        ffn_tensor_parallel=args.ffn_tensor_parallel,
                        deployment_mode=args.deployment_mode,
                        device_type2=args.device_type2,
                        min_die2=args.min_die2,
                        max_die2=args.max_die2,
                        die_step2=args.die_step2
                    )
                    deepep_search = DeepEpSearch(config)
                    deepep_search.deployment()
            else:
                # Direct calculation mode
                config = Config(
                    serving_mode=args.serving_mode,
                    model_type=args.model_type,
                    device_type=args.device_type,
                    min_attn_bs=args.min_attn_bs,
                    max_attn_bs=args.max_attn_bs,
                    min_die=args.min_die,
                    max_die=args.max_die,
                    die_step=args.die_step,
                    tpot=None,
                    attn_bs=args.attn_bs,
                    kv_len=kv_len,
                    micro_batch_num=1,
                    next_n=args.next_n,
                    multi_token_ratio=args.multi_token_ratio,
                    attn_tensor_parallel=args.attn_tensor_parallel,
                    ffn_tensor_parallel=args.ffn_tensor_parallel,
                    deployment_mode=args.deployment_mode,
                    device_type2=args.device_type2,
                    min_die2=args.min_die2,
                    max_die2=args.max_die2,
                    die_step2=args.die_step2
                )
                deepep_search = DeepEpSearch(config)
                deepep_search.deployment()
    else:
        raise ValueError(f"Invalid serving mode: {args.serving_mode}")
```

### Step 1.4: Update post-plot generation logic

Modify the post-plot generation section (lines 140-184) to handle both modes:

```python
    if os.environ.get("LIGHT_LLM_SKIP_POST_PLOTS", "").lower() in ("1", "true", "yes"):
        return

    import subprocess
    throughput_cmd = [
        "python", "src/visualization/throughput.py",
        "--model_type", args.model_type,
        "--device_type", args.device_type,
        "--deployment_mode", args.deployment_mode,
        "--min_die", str(args.min_die),
        "--max_die", str(args.max_die),
    ]
    # Add heterogeneous parameters if applicable
    if args.deployment_mode == "Heterogeneous":
        if args.device_type2 is None:
                raise ValueError("device_type2 is required for Heterogeneous deployment mode")
        throughput_cmd.extend(["--device_type2", args.device_type2])
        _min_die2 = args.min_die2 if args.min_die2 is not None else args.min_die
        _max_die2 = args.max_die2 if args.max_die2 is not None else args.max_die
        _die_step2 = args.die_step2 if args.die_step2 is not None else args.die_step
        total_dies = list(range(args.min_die + _min_die2, args.max_die + _max_die2 + 1, min(args.die_step, _die_step2)))
    else:
        total_dies = list(range(args.min_die, args.max_die + 1, args.die_step))
    
    throughput_cmd.extend(["--micro_batch_num", "2", "3"])
    
    # Handle mode-specific parameters
    if mode == "constraint" and args.tpot:
        throughput_cmd.extend(["--tpot_list"] + [str(t) for t in args.tpot])
    elif mode == "direct" and args.attn_bs:
        throughput_cmd.extend(["--attn_bs_list"] + [str(bs) for bs in args.attn_bs])
    
    throughput_cmd.extend(["--kv_len_list"] + [str(k) for k in args.kv_len])
    throughput_cmd.extend(["--total_die"] + [str(t) for t in total_dies])
    subprocess.run(throughput_cmd)

    # Pipeline visualization - only for constraint mode or specific attn_bs in direct mode
    if mode == "constraint":
        for tpot in args.tpot:
            for kv_len in args.kv_len:
                # Construct file name based on deployment mode
                if args.deployment_mode == "Heterogeneous":
                    device_type2_enum = DeviceType(args.device_type2)
                    file_name = f"{DeviceType(args.device_type).name}_{device_type2_enum.name}-{ModelType(args.model_type).name}-tpot{tpot}-kv_len{kv_len}.csv"
                else:
                    file_name = f"{DeviceType(args.device_type).name}-{ModelType(args.model_type).name}-tpot{tpot}-kv_len{kv_len}.csv"
                pipeline_cmd = [
                    "python", "src/visualization/pipeline.py",
                    "--file_name", file_name,
                    "--deployment_mode", args.deployment_mode,
                ]
                if args.deployment_mode == "Heterogeneous":
                    pipeline_cmd.extend(["--device_type2", args.device_type2])
                subprocess.run(pipeline_cmd)
    else:
        # Direct mode - generate pipeline for each attn_bs
        for attn_bs in args.attn_bs:
            for kv_len in args.kv_len:
                if args.deployment_mode == "Heterogeneous":
                    device_type2_enum = DeviceType(args.device_type2)
                    file_name = f"{DeviceType(args.device_type).name}_{device_type2_enum.name}-{ModelType(args.model_type).name}-bs{attn_bs}-kv_len{kv_len}.csv"
                else:
                    file_name = f"{DeviceType(args.device_type).name}-{ModelType(args.model_type).name}-bs{attn_bs}-kv_len{kv_len}.csv"
                pipeline_cmd = [
                    "python", "src/visualization/pipeline.py",
                    "--file_name", file_name,
                    "--deployment_mode", args.deployment_mode,
                ]
                if args.deployment_mode == "Heterogeneous":
                    pipeline_cmd.extend(["--device_type2", args.device_type2])
                subprocess.run(pipeline_cmd)
```

### Step 1.5: Commit CLI changes

```bash
git add src/cli/main.py
git commit -m "feat(cli): add attn_bs parameter and dual-mode detection for direct calculation"
```

---

## Task 2: Config Class Changes

**Files:**
- Modify: `conf/config.py:21-132`

### Step 2.1: Add attn_bs parameter to __init__

Add `attn_bs` parameter to the `__init__` signature (line 21), insert after `tpot`:

```python
def __init__(
    self,
    serving_mode: str,
    model_type: ModelType,
    device_type: DeviceType,
    min_attn_bs: int,
    max_attn_bs: int,
    min_die: int,
    max_die: int,
    die_step: int,
    tpot: Optional[list[int]],  # Changed: can be None now
    attn_bs: Optional[list[int]],  # New parameter
    kv_len: list[int],
    micro_batch_num: list[int],
    next_n: int,
    multi_token_ratio: float,
    attn_tensor_parallel: int,
    ffn_tensor_parallel: int,
    deployment_mode: str = "Homogeneous",
    device_type2: Optional[str] = None,
    min_die2: Optional[int] = None,
    max_die2: Optional[int] = None,
    die_step2: Optional[int] = None
) -> None:
```

### Step 2.2: Add docstring for attn_bs parameter

Add docstring for the new `attn_bs` parameter (around line 65):

```python
Args:
    ...
    tpot: The target TPOT for constraint mode. None for direct calculation mode.
    attn_bs: The attention batch size list for direct calculation mode. Required when tpot is None.
    ...
```

### Step 2.3: Add mode-aware initialization logic

Add the conditional initialization logic after line 118 (after `self.die_step2 = die_step`):

```python
        # Mode-aware initialization
        if tpot is not None and len(tpot) > 0:
            # Constraint mode
            self.tpot = tpot
            self.attn_bs = None
            self.mode = "constraint"
        else:
            # Direct calculation mode
            self.tpot = None
            if attn_bs is None or len(attn_bs) == 0:
                raise ValueError("attn_bs must be specified when tpot is not provided")
            self.attn_bs = attn_bs
            self.mode = "direct"
```

### Step 2.4: Modify tpot attribute assignment

Change the direct assignment of `self.tpot` (line 119) to be conditional:

```python
        # Remove: self.tpot = tpot
        # The conditional assignment is now handled in Step 2.3
```

### Step 2.5: Commit Config changes

```bash
git add conf/config.py
git commit -m "feat(config): add attn_bs attribute and mode-aware initialization"
```

---

## Task 3: AfdSearch Direct Calculation Method

**Files:**
- Modify: `src/search/afd.py:1-296`

### Step 3.1: Rename existing search method

Rename `search` method to `search_with_constraint` (line 154):

```python
    def search_with_constraint(self):
        '''
        Description:
            Search the optimal attention batch size,
            attention die count, FFN die count for the model used AFD serving.
            For each (ffn_die, attn_die) pair, binary search for the max attn_bs
            that satisfies latency and memory constraints.
        '''
        # ... existing implementation unchanged
```

### Step 3.2: Add _evaluate_config_direct method

Add new method `_evaluate_config_direct` after `_evaluate_config` method (around line 153). This method evaluates configuration without TPOT constraint:

```python
    def _evaluate_config_direct(self, attn_bs, attn_die, ffn_die, routed_expert_per_die):
        """Evaluate a (attn_bs, attn_die, ffn_die) configuration without TPOT constraint.
        Returns results dict if memory constraints are satisfied, None otherwise.
        """
        # Memory computation
        if get_attention_family(self.config.model_type) == "MLA":
            kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(
                self.config.model_config, attn_bs)
        elif get_attention_family(self.config.model_type) == "GQA":
            kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(
                self.config.model_config, attn_bs)

        ffn_static_memory = per_router_expert_memory * routed_expert_per_die

        # Memory constraints only
        attn_used_memory = kv_size * self.config.micro_batch_num + attn_static_memory
        attn_memory_threshold = self.config.aichip_config_attn.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO
        ffn_memory_threshold = self.config.aichip_config_ffn.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO

        if attn_used_memory > attn_memory_threshold or ffn_static_memory > ffn_memory_threshold:
            return None

        # Compute MoE layer timing
        self.config.attn_bs = attn_bs
        self.config.ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok * attn_die / ffn_die
        moe_ffn_bs = self.config.ffn_bs
        self.config.attn_die = attn_die
        self.config.ffn_die = ffn_die
        self.config.routed_expert_per_die = routed_expert_per_die

        model = get_model(self.config)
        attn = model["attn"]
        attn()
        moe = model["moe"]
        moe()

        attn_time = attn.e2e_time * SEC_2_US
        moe_time = moe.e2e_time * SEC_2_US
        commu_time = moe.commu_time * SEC_2_US

        e2e_time_per_moe_layer = max(
            attn_time + moe_time + commu_time,
            max(attn_time, max(moe_time, commu_time)) * self.config.micro_batch_num
        )

        e2e_time = e2e_time_per_moe_layer * (self.config.model_config.num_moe_layers + self.config.seq_len - 1)
        embedding = model["embedding"]
        embedding()
        lm_head = model["lm_head"]
        lm_head()
        e2e_time = e2e_time + embedding.e2e_time * SEC_2_US + lm_head.e2e_time * SEC_2_US

        # Dense layer computation
        if self.config.model_config.num_layers > self.config.model_config.num_moe_layers:
            self.config.attn_bs = attn_bs * self.config.micro_batch_num
            self.config.ffn_bs = self.config.attn_bs
            dense_ffn_bs = self.config.ffn_bs
            model_dense = get_model(self.config)
            attn_dense = model_dense["attn"]
            mlp = model_dense["mlp"]
            attn_dense()
            mlp()
            e2e_time_per_dense_layer = attn_dense.e2e_time * SEC_2_US + mlp.e2e_time * SEC_2_US
            e2e_time = e2e_time + e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace
        else:
            dense_ffn_bs = 0
            e2e_time_per_dense_layer = 0.0

        # NO TPOT constraint check - removed

        e2e_time = e2e_time / (1 + self.config.multi_token_ratio) * US_2_MS

        a2e_send = attn.a2e_send.e2e_time * SEC_2_US
        a2e_recv = attn.a2e_recv.e2e_time * SEC_2_US
        e2a_recv = moe.e2a_recv.e2e_time * SEC_2_US
        dispatch_time = moe.dispatch_time * SEC_2_US
        combine_time = moe.combine_time * SEC_2_US

        ffn_used_memory = ffn_static_memory
        attn_available_memory = attn_memory_threshold - attn_used_memory
        ffn_available_memory = ffn_memory_threshold - ffn_used_memory

        return {
            'moe_ffn_bs': moe_ffn_bs,
            'dense_ffn_bs': dense_ffn_bs,
            'attn_time': attn_time,
            'moe_time': moe_time,
            'dispatch_time': dispatch_time,
            'combine_time': combine_time,
            'commu_time': commu_time,
            'e2e_time': e2e_time,
            'e2e_time_per_dense_layer': e2e_time_per_dense_layer,
            'e2e_time_per_moe_layer': e2e_time_per_moe_layer,
            'a2e_send': a2e_send,
            'a2e_recv': a2e_recv,
            'e2a_recv': e2a_recv,
            'kv_size': kv_size,
            'attn_static_memory': attn_static_memory,
            'mlp_static_memory': mlp_static_memory,
            'ffn_static_memory': ffn_static_memory,
            'attn_used_memory': attn_used_memory,
            'ffn_used_memory': ffn_used_memory,
            'attn_available_memory': attn_available_memory,
            'ffn_available_memory': ffn_available_memory,
        }
```

### Step 3.3: Add search_direct method

Add new method `search_direct` after `search_with_constraint` method:

```python
    def search_direct(self):
        '''
        Description:
            Direct calculation mode: iterate over attn_bs list and compute performance
            for each (attn_bs, attn_die, ffn_die) configuration.
            Only memory constraints are checked, no TPOT constraint.
        '''

        # Get die ranges based on deployment mode
        if self.config.deployment_mode == "Heterogeneous":
            min_ffn_die = self.config.min_die2
            max_ffn_die = self.config.max_die2 + 1
            ffn_die_step = self.config.die_step2
            min_attn_die = self.config.min_die
            max_attn_die = self.config.max_die + 1
            attn_die_step = self.config.die_step
        else:
            min_ffn_die = self.config.min_die
            max_ffn_die = self.config.max_die + 1
            ffn_die_step = self.config.die_step
            min_attn_die = None
            max_attn_die = None
            attn_die_step = self.config.die_step

        for ffn_die in range(min_ffn_die, max_ffn_die, ffn_die_step):
            routed_expert_per_die = Config.calc_routed_expert_per_die(
                self.config.model_config.n_routed_experts,
                self.config.model_config.n_shared_experts,
                ffn_die
            )

            # Determine attention die range
            if self.config.deployment_mode == "Heterogeneous":
                attn_die_start = min_attn_die
                attn_die_end = max_attn_die
            else:
                attn_die_start = ffn_die
                attn_die_end = self.ATTN_DIE_MULTIPLIER * ffn_die

            for attn_die in range(attn_die_start, attn_die_end, attn_die_step):
                total_die = ffn_die + attn_die

                # Node alignment check based on deployment mode
                if self.config.deployment_mode == "Heterogeneous":
                    if attn_die % self.config.aichip_config_attn.num_dies_per_node != 0:
                        continue
                    if ffn_die % self.config.aichip_config_ffn.num_dies_per_node != 0:
                        continue
                else:
                    if total_die % self.config.aichip_config.num_dies_per_node != 0:
                        continue

                # Iterate over attn_bs list
                for attn_bs in self.config.attn_bs:
                    result = self._evaluate_config_direct(
                        attn_bs, attn_die, ffn_die, routed_expert_per_die
                    )
                    if result is None:
                        continue  # Memory constraint failed

                    throughput = (
                        attn_bs * self.config.micro_batch_num * attn_die / total_die / result['e2e_time'] / MS_2_SEC
                    )

                    logging.info(f"-------AFD Direct Calculation Result:-------")
                    logging.info(
                        f"deployment_mode: {self.config.deployment_mode}, "
                        f"attn_bs: {attn_bs}, dense_ffn_bs: {result['dense_ffn_bs']}, moe_ffn_bs: {result['moe_ffn_bs']}, "
                        f"kv_len: {self.config.kv_len}, attn_die: {attn_die}, "
                        f"ffn_die: {ffn_die}, total_die: {total_die}, "
                        f"attn_time: {result['attn_time']:.2f}us, moe_time: {result['moe_time']:.2f}us, "
                        f"a2e_send: {result['a2e_send']:.2f}us, a2e_recv: {result['a2e_recv']:.2f}us, "
                        f"dispatch_time: {result['dispatch_time']:.2f}us, combine_time: {result['combine_time']:.2f}us, "
                        f"e2a_recv: {result['e2a_recv']:.2f}us, commu_time: {result['commu_time']:.2f}us, e2e_time: {result['e2e_time']:.2f}ms, "
                        f"e2e_time_per_dense_layer: {result['e2e_time_per_dense_layer']:.2f}us, "
                        f"e2e_time_per_moe_layer: {result['e2e_time_per_moe_layer']:.2f}us, throughput: {throughput:.2f} tokens/die/s, "
                        f"kv_size: {result['kv_size']} GB, attn_static_memory: {result['attn_static_memory']} GB, "
                        f"mlp_static_memory: {result['mlp_static_memory']} GB, ffn_static_memory: {result['ffn_static_memory']} GB, "
                        f"attn_used_memory: {result['attn_used_memory']} GB, ffn_used_memory: {result['ffn_used_memory']} GB, "
                        f"attn_available_memory: {result['attn_available_memory']} GB, ffn_available_memory: {result['ffn_available_memory']} GB"
                    )

                    self.perf_afd_results.append([
                        attn_bs, result['dense_ffn_bs'], result['moe_ffn_bs'], self.config.kv_len, attn_die, ffn_die, total_die,
                        result['attn_time'], result['moe_time'], result['a2e_send'], result['a2e_recv'],
                        result['dispatch_time'], result['combine_time'], result['e2a_recv'], result['commu_time'],
                        result['e2e_time'], result['e2e_time_per_dense_layer'], result['e2e_time_per_moe_layer'], throughput,
                        result['kv_size'], result['attn_static_memory'], result['mlp_static_memory'], result['ffn_static_memory'],
                        result['attn_used_memory'], result['ffn_used_memory'],
                        result['attn_available_memory'], result['ffn_available_memory'],
                        self.config.deployment_mode, self.config.device_type.name, self.config.device_type2.name
                    ])

        columns = [
            'attn_bs(per_micro_batch)', 'dense_ffn_bs', 'moe_ffn_bs(per_micro_batch)',
            'kv_len', 'attn_die', 'ffn_die', 'total_die',
            'attn_time(us)', 'moe_time(us)', 'a2e_send(us)', 'a2e_recv(us)', 'dispatch_time(us)', 'combine_time(us)', 'e2a_recv(us)', 'commu_time(us)',
            'e2e_time(ms)', 'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)', 'throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)',
            'attn_used_memory(GB)', 'ffn_used_memory(GB)', 'attn_available_memory(GB)', 'ffn_available_memory(GB)',
            'deployment_mode', 'device_type_attn', 'device_type_ffn'
        ]
        df = pd.DataFrame(self.perf_afd_results, columns=columns)

        # Generate file name and directory based on deployment mode
        # For direct mode, use bs{attn_bs} instead of tpot{tpot}
        # Since we iterate over multiple attn_bs values, we need to group by attn_bs
        # For simplicity, we output a combined file with all results
        # But per design, we output separate files for each attn_bs
        
        # Group by attn_bs and save separate files
        for attn_bs_val in self.config.attn_bs:
            df_bs = df[df['attn_bs(per_micro_batch)'] == attn_bs_val]
            if len(df_bs) == 0:
                continue
                
            if self.config.deployment_mode == "Heterogeneous":
                result_dir = f"data/afd/mbn{self.config.micro_batch_num}/heterogeneous/"
                file_name = f"{self.config.device_type.name}_{self.config.device_type2.name}-{self.config.model_type.name}-bs{attn_bs_val}-kv_len{self.config.kv_len}.csv"
            else:
                result_dir = f"data/afd/mbn{self.config.micro_batch_num}/homogeneous/"
                file_name = f"{self.config.device_type.name}-{self.config.model_type.name}-bs{attn_bs_val}-kv_len{self.config.kv_len}.csv"
            os.makedirs(result_dir, exist_ok=True)
            result_path = result_dir + file_name
            df_bs.to_csv(result_path, index=False, float_format='%.2f')

            if len(df_bs) > 0:
                df_best = df_bs.sort_values(by=['throughput(tokens/die/s)'], ascending=False).drop_duplicates(subset=['total_die'])
                df_best = df_best.sort_values(by=['total_die'], ascending=True)
                if self.config.deployment_mode == "Heterogeneous":
                    best_result_dir = f"data/afd/mbn{self.config.micro_batch_num}/best/heterogeneous/"
                else:
                    best_result_dir = f"data/afd/mbn{self.config.micro_batch_num}/best/homogeneous/"
                os.makedirs(best_result_dir, exist_ok=True)
                best_result_path = best_result_dir + file_name
                df_best.to_csv(best_result_path, index=False, float_format='%.2f')
```

### Step 3.4: Add dispatcher in deployment method

Modify the `deployment` method (line 295) to dispatch based on mode:

```python
    def deployment(self):
        if self.config.mode == "constraint":
            self.search_with_constraint()
        else:
            self.search_direct()
```

### Step 3.5: Update file naming in search_with_constraint

Ensure the existing `search_with_constraint` method still uses `tpot` in filename. No changes needed as the existing implementation already does this correctly (lines 274-282).

### Step 3.6: Commit AfdSearch changes

```bash
git add src/search/afd.py
git commit -m "feat(afd): add search_direct method for direct calculation mode without TPOT constraint"
```

---

## Task 4: DeepEP Search Direct Calculation Method

**Files:**
- Modify: `src/search/deepep.py:1-297`

### Step 4.1: Add _evaluate_config_direct method

Add new method `_evaluate_config_direct` after existing `_evaluate_config` method (around line 107):

```python
    def _evaluate_config_direct(self, attn_bs, temp_config, routed_expert_per_die):
        """Evaluate a single (attn_bs) configuration without TPOT constraint.

        Returns:
            dict with all result fields, or None if memory constraints are violated.
        """
        if get_attention_family(self.config.model_type) == "MLA":
            kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_MLA_memory_size(
                self.config.model_config, attn_bs
            )
        elif get_attention_family(self.config.model_type) == "GQA":
            kv_size, attn_static_memory, mlp_static_memory, per_router_expert_memory = self.compute_GQA_memory_size(
                self.config.model_config, attn_bs
            )
        ffn_bs = attn_bs * self.config.model_config.num_experts_per_tok
        temp_config.attn_bs = attn_bs
        temp_config.ffn_bs = ffn_bs

        model = get_model(temp_config)
        attn = model["attn"]
        attn()
        moe = model["moe"]
        moe()
        attn_time = attn.e2e_time * SEC_2_US
        moe_time = moe.e2e_time * SEC_2_US
        commu_time = moe.commu_time * SEC_2_US
        dispatch_time = moe.dispatch_time * SEC_2_US
        combine_time = moe.combine_time * SEC_2_US

        ffn_dynamic_memory = (
            ffn_bs * self.config.model_config.hidden_size *
            self.config.model_config.num_layers * BYTE_2_GB
        )
        ffn_static_memory = per_router_expert_memory * routed_expert_per_die
        used_memory = (
            kv_size * self.config.micro_batch_num + attn_static_memory +
            mlp_static_memory + ffn_dynamic_memory + ffn_static_memory
        )
        
        # Memory constraint check only
        from conf.hardware_config import HWConf
        aichip_config = HWConf.create(temp_config.device_type)
        if used_memory > aichip_config.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO:
            return None
        
        e2e_time_per_moe_layer = attn_time + moe_time + commu_time
        e2e_time = e2e_time_per_moe_layer * (self.config.model_config.num_moe_layers + self.config.seq_len - 1)
        embedding = model["embedding"]
        embedding()
        lm_head = model["lm_head"]
        lm_head()
        e2e_time = e2e_time + embedding.e2e_time * SEC_2_US + lm_head.e2e_time * SEC_2_US

        if self.config.model_config.num_layers > self.config.model_config.num_moe_layers:
            mlp = model["mlp"]
            mlp()
            mlp_time = mlp.e2e_time * SEC_2_US
            e2e_time_per_dense_layer = attn_time + mlp_time
            e2e_time = e2e_time + e2e_time_per_dense_layer * self.config.model_config.first_k_dense_replace
        else:
            e2e_time_per_dense_layer = 0.0

        e2e_time = e2e_time / (1 + self.config.multi_token_ratio) * US_2_MS

        available_memory = aichip_config.aichip_memory * BYTE_2_GB * MEMORY_THRESHOLD_RATIO - used_memory

        return {
            'ffn_bs': ffn_bs,
            'attn_time': attn_time,
            'moe_time': moe_time,
            'dispatch_time': dispatch_time,
            'combine_time': combine_time,
            'commu_time': commu_time,
            'e2e_time': e2e_time,
            'e2e_time_per_dense_layer': e2e_time_per_dense_layer,
            'e2e_time_per_moe_layer': e2e_time_per_moe_layer,
            'kv_size': kv_size,
            'attn_static_memory': attn_static_memory,
            'mlp_static_memory': mlp_static_memory,
            'ffn_static_memory': ffn_static_memory,
            'used_memory': used_memory,
            'available_memory': available_memory,
        }
```

### Step 4.2: Add search_bs_direct method

Add new method `search_bs_direct` after existing `search_bs` method (around line 286):

```python
    def search_bs_direct(self):
        '''
        Description:
            Direct calculation mode: iterate over attn_bs list and compute performance
            for each (attn_bs, total_die) configuration.
            Only memory constraints are checked, no TPOT constraint.
        '''
        from conf.hardware_config import DeviceType, HWConf

        device_type = self.config.device_type
        aichip_config = HWConf.create(device_type)

        for total_die in range(self.config.min_die, self.config.max_die + 1, self.config.die_step):
            routed_expert_per_die = Config.calc_routed_expert_per_die(
                self.config.model_config.n_routed_experts,
                self.config.model_config.n_shared_experts,
                total_die
            )

            # Create a temporary config
            temp_config = Config(
                serving_mode="DeepEP",
                model_type=self.config.model_type.value,
                device_type=self.config.device_type.value,
                min_attn_bs=self.config.min_attn_bs,
                max_attn_bs=self.config.max_attn_bs,
                min_die=self.config.min_die,
                max_die=self.config.max_die,
                die_step=self.config.die_step,
                tpot=None,
                attn_bs=self.config.attn_bs,
                kv_len=self.config.kv_len,
                micro_batch_num=1,
                next_n=self.config.seq_len - 1,
                multi_token_ratio=self.config.multi_token_ratio,
                attn_tensor_parallel=self.config.attn_tensor_parallel,
                ffn_tensor_parallel=self.config.ffn_tensor_parallel,
                deployment_mode="Homogeneous"
            )
            temp_config.attn_die = total_die
            temp_config.ffn_die = total_die
            temp_config.routed_expert_per_die = routed_expert_per_die

            # Iterate over attn_bs list
            for attn_bs in self.config.attn_bs:
                r = self._evaluate_config_direct(attn_bs, temp_config, routed_expert_per_die)
                if r is None:
                    continue  # Memory constraint failed

                r['attn_bs'] = attn_bs
                r['throughput'] = attn_bs / r['e2e_time'] / MS_2_SEC

                logging.info(f"-------DeepEP Direct Calculation Result:-------")
                logging.info(
                    f"total_die: {total_die}, attn_bs: {attn_bs}, ffn_bs: {r['ffn_bs']}, "
                    f"attn_time: {r['attn_time']:.2f}us, moe_time: {r['moe_time']:.2f}us, "
                    f"e2e_time: {r['e2e_time']:.2f}ms, throughput: {r['throughput']:.2f} tokens/die/s"
                )

                self.perf_deepep_results.append([
                    r['attn_bs'], r['ffn_bs'], self.config.kv_len, total_die,
                    r['attn_time'], r['moe_time'], r['dispatch_time'], r['combine_time'], r['commu_time'], r['e2e_time'],
                    r['e2e_time_per_dense_layer'], r['e2e_time_per_moe_layer'], r['throughput'],
                    r['kv_size'], r['attn_static_memory'], r['mlp_static_memory'], r['ffn_static_memory'],
                    r['used_memory'], r['available_memory'],
                    "Homogeneous", self.config.device_type.name, self.config.device_type.name
                ])

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'total_die',
            'attn_time(us)', 'moe_time(us)', 'dispatch_time(us)', 'combine_time(us)', 'commu_time(us)', 'e2e_time(ms)',
            'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)', 'throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)',
            'used_memory(GB)', 'available_memory(GB)',
            'deployment_mode', 'device_type1', 'device_type2'
        ]
        df = pd.DataFrame(self.perf_deepep_results, columns=columns)

        # Save separate files for each attn_bs
        for attn_bs_val in self.config.attn_bs:
            df_bs = df[df['attn_bs'] == attn_bs_val]
            if len(df_bs) == 0:
                continue

            result_dir = f"data/deepep/homogeneous/"
            file_name = f"{self.config.device_type.name}-{self.config.model_type.name}-bs{attn_bs_val}-kv_len{self.config.kv_len}.csv"
            os.makedirs(result_dir, exist_ok=True)
            result_path = result_dir + file_name
            df_bs.to_csv(result_path, index=False, float_format='%.2f')
```

### Step 4.3: Add search_bs_heterogeneous_direct method

Add new method `search_bs_heterogeneous_direct` after `search_bs_heterogeneous` method:

```python
    def search_bs_heterogeneous_direct(self):
        '''
        Run heterogeneous DeepEP direct calculation mode.
        Iterate over attn_bs for both device types and compute weighted average throughput.
        '''
        logging.info("=" * 60)
        logging.info("NOTE: DeepEP Heterogeneous direct calculation mode runs")
        logging.info("direct calculation on two device types separately.")
        logging.info("=" * 60)

        from conf.hardware_config import DeviceType, HWConf

        results_device1 = {}
        results_device2 = {}

        # Run direct calculation on device1
        for total_die in range(self.config.min_die, self.config.max_die + 1, self.config.die_step):
            routed_expert_per_die = Config.calc_routed_expert_per_die(
                self.config.model_config.n_routed_experts,
                self.config.model_config.n_shared_experts,
                total_die
            )

            temp_config = Config(
                serving_mode="DeepEP",
                model_type=self.config.model_type.value,
                device_type=self.config.device_type.value,
                min_attn_bs=self.config.min_attn_bs,
                max_attn_bs=self.config.max_attn_bs,
                min_die=self.config.min_die,
                max_die=self.config.max_die,
                die_step=self.config.die_step,
                tpot=None,
                attn_bs=self.config.attn_bs,
                kv_len=self.config.kv_len,
                micro_batch_num=1,
                next_n=self.config.seq_len - 1,
                multi_token_ratio=self.config.multi_token_ratio,
                attn_tensor_parallel=self.config.attn_tensor_parallel,
                ffn_tensor_parallel=self.config.ffn_tensor_parallel,
                deployment_mode="Homogeneous"
            )
            temp_config.attn_die = total_die
            temp_config.ffn_die = total_die
            temp_config.routed_expert_per_die = routed_expert_per_die

            results_device1[total_die] = {}
            for attn_bs in self.config.attn_bs:
                r = self._evaluate_config_direct(attn_bs, temp_config, routed_expert_per_die)
                if r is None:
                    continue
                r['attn_bs'] = attn_bs
                r['throughput'] = attn_bs / r['e2e_time'] / MS_2_SEC
                results_device1[total_die][attn_bs] = r

        # Run direct calculation on device2
        for total_die in range(self.config.min_die2, self.config.max_die2 + 1, self.config.die_step2):
            routed_expert_per_die = Config.calc_routed_expert_per_die(
                self.config.model_config.n_routed_experts,
                self.config.model_config.n_shared_experts,
                total_die
            )

            temp_config = Config(
                serving_mode="DeepEP",
                model_type=self.config.model_type.value,
                device_type=self.config.device_type2.value,
                min_attn_bs=self.config.min_attn_bs,
                max_attn_bs=self.config.max_attn_bs,
                min_die=self.config.min_die2,
                max_die=self.config.max_die2,
                die_step=self.config.die_step2,
                tpot=None,
                attn_bs=self.config.attn_bs,
                kv_len=self.config.kv_len,
                micro_batch_num=1,
                next_n=self.config.seq_len - 1,
                multi_token_ratio=self.config.multi_token_ratio,
                attn_tensor_parallel=self.config.attn_tensor_parallel,
                ffn_tensor_parallel=self.config.ffn_tensor_parallel,
                deployment_mode="Homogeneous"
            )
            temp_config.attn_die = total_die
            temp_config.ffn_die = total_die
            temp_config.routed_expert_per_die = routed_expert_per_die

            results_device2[total_die] = {}
            for attn_bs in self.config.attn_bs:
                r = self._evaluate_config_direct(attn_bs, temp_config, routed_expert_per_die)
                if r is None:
                    continue
                r['attn_bs'] = attn_bs
                r['throughput'] = attn_bs / r['e2e_time'] / MS_2_SEC
                results_device2[total_die][attn_bs] = r

        # Combine results with weighted average throughput for each attn_bs
        for die1, r1_by_bs in results_device1.items():
            for die2, r2_by_bs in results_device2.items():
                total_die = die1 + die2
                for attn_bs in self.config.attn_bs:
                    if attn_bs not in r1_by_bs or attn_bs not in r2_by_bs:
                        continue
                    
                    r1 = r1_by_bs[attn_bs]
                    r2 = r2_by_bs[attn_bs]
                    weighted_throughput = (die1 * r1['throughput'] + die2 * r2['throughput']) / total_die

                    logging.info(f"-------DeepEP Heterogeneous Direct Result:-------")
                    logging.info(
                        f"device1_die:{die1}, device2_die:{die2}, total_die:{total_die}, attn_bs:{attn_bs}, "
                        f"throughput_device1:{r1['throughput']:.2f}, throughput_device2:{r2['throughput']:.2f}, "
                        f"weighted_throughput:{weighted_throughput:.2f} tokens/die/s"
                    )

                    self.perf_deepep_results.append([
                        r1['attn_bs'], r1['ffn_bs'], self.config.kv_len, die1, die2, total_die,
                        r1['attn_time'], r1['moe_time'], r1['dispatch_time'], r1['combine_time'], r1['commu_time'], r1['e2e_time'],
                        r1['e2e_time_per_dense_layer'], r1['e2e_time_per_moe_layer'],
                        r1['throughput'], r2['throughput'], weighted_throughput,
                        r1['kv_size'], r1['attn_static_memory'], r1['mlp_static_memory'], r1['ffn_static_memory'],
                        r1['used_memory'], r1['available_memory'],
                        "Heterogeneous", self.config.device_type.name, self.config.device_type2.name
                    ])

        columns = [
            'attn_bs', 'ffn_bs', 'kv_len', 'device1_die', 'device2_die', 'total_die',
            'attn_time(us)', 'moe_time(us)', 'dispatch_time(us)', 'combine_time(us)', 'commu_time(us)', 'e2e_time(ms)',
            'e2e_time_per_dense_layer(us)', 'e2e_time_per_moe_layer(us)',
            'throughput_device1(tokens/die/s)', 'throughput_device2(tokens/die/s)', 'weighted_throughput(tokens/die/s)',
            'kv_size(GB)', 'attn_static_memory(GB)', 'mlp_static_memory(GB)', 'ffn_static_memory(GB)',
            'used_memory(GB)', 'available_memory(GB)',
            'deployment_mode', 'device_type1', 'device_type2'
        ]
        df = pd.DataFrame(self.perf_deepep_results, columns=columns)

        # Save separate files for each attn_bs
        for attn_bs_val in self.config.attn_bs:
            df_bs = df[df['attn_bs'] == attn_bs_val]
            if len(df_bs) == 0:
                continue

            result_dir = f"data/deepep/heterogeneous/"
            file_name = f"{self.config.device_type.name}_{self.config.device_type2.name}-{self.config.model_type.name}-bs{attn_bs_val}-kv_len{self.config.kv_len}.csv"
            os.makedirs(result_dir, exist_ok=True)
            result_path = result_dir + file_name
            df_bs.to_csv(result_path, index=False, float_format='%.2f')
```

### Step 4.4: Modify deployment method dispatcher

Modify the `deployment` method (line 288) to dispatch based on mode:

```python
    def deployment(self):
        '''
        Run DeepEP deployment based on deployment_mode and calculation mode.
        '''
        if self.config.mode == "constraint":
            if self.config.deployment_mode == "Heterogeneous":
                self.search_bs_heterogeneous()
            else:
                self.search_bs()
        else:
            # Direct calculation mode
            if self.config.deployment_mode == "Heterogeneous":
                self.search_bs_heterogeneous_direct()
            else:
                self.search_bs_direct()
```

### Step 4.5: Commit DeepEP changes

```bash
git add src/search/deepep.py
git commit -m "feat(deepep): add search_bs_direct and search_bs_heterogeneous_direct for direct calculation mode"
```

---

## Task 5: Webapp Backend API Changes

**Files:**
- Modify: `webapp/backend/main.py:1-313`

### Step 5.1: Add attn_bs to RunRequest model

Modify `RunRequest` class (lines 36-57) to add `attn_bs` field and make `tpot` optional:

```python
class RunRequest(BaseModel):
    serving_mode: str = "AFD"
    model_type: str = "deepseek-ai/DeepSeek-V3"
    device_type: str = "Ascend_A3Pod"
    min_attn_bs: int = 2
    max_attn_bs: int = 1000
    min_die: int = 16
    max_die: int = 768
    die_step: Optional[int] = 16
    tpot: Optional[List[int]] = None  # Changed: default None, optional
    attn_bs: Optional[List[int]] = None  # New: batchsize list for direct mode
    kv_len: Optional[List[int]] = [4096]
    micro_batch_num: Optional[List[int]] = [2]
    next_n: int = 1
    multi_token_ratio: float = 0.7
    attn_tensor_parallel: int = 1
    ffn_tensor_parallel: int = 1
    # Heterogeneous mode parameters
    deployment_mode: str = "Homogeneous"
    device_type2: Optional[str] = None
    min_die2: Optional[int] = None
    max_die2: Optional[int] = None
    die_step2: Optional[int] = None
```

### Step 5.2: Modify _sanitize_args function

Modify `_sanitize_args` function (lines 60-94) to handle both tpot and attn_bs:

```python
def _sanitize_args(req: RunRequest) -> List[str]:
    args = [
        "python", SRC_CLI,
        "--serving_mode", req.serving_mode,
        "--model_type", req.model_type,
        "--device_type", req.device_type,
        "--min_attn_bs", str(req.min_attn_bs),
        "--max_attn_bs", str(req.max_attn_bs),
        "--min_die", str(req.min_die),
        "--max_die", str(req.max_die),
        "--next_n", str(req.next_n),
        "--multi_token_ratio", str(req.multi_token_ratio),
        "--attn_tensor_parallel", str(req.attn_tensor_parallel),
        "--ffn_tensor_parallel", str(req.ffn_tensor_parallel),
        "--deployment_mode", req.deployment_mode,
    ]
    if req.die_step is not None:
        args += ["--die_step", str(req.die_step)]
    # Handle mode parameters
    if req.tpot:
        args += ["--tpot"] + [str(x) for x in req.tpot]
    if req.attn_bs:
        args += ["--attn_bs"] + [str(x) for x in req.attn_bs]
    if req.kv_len:
        args += ["--kv_len"] + [str(x) for x in req.kv_len]
    if req.micro_batch_num:
        args += ["--micro_batch_num"] + [str(x) for x in req.micro_batch_num]
    # Heterogeneous mode parameters
    if req.deployment_mode == "Heterogeneous":
        if req.device_type2:
            args += ["--device_type2", req.device_type2]
        if req.min_die2 is not None:
            args += ["--min_die2", str(req.min_die2)]
        if req.max_die2 is not None:
            args += ["--max_die2", str(req.max_die2)]
        if req.die_step2 is not None:
            args += ["--die_step2", str(req.die_step2)]
    return args
```

### Step 5.3: Add validation in start_run endpoint

Modify `start_run` endpoint (lines 106-113) to validate mode parameters:

```python
@api.post("/run")
def start_run(req: RunRequest, background_tasks: BackgroundTasks):
    # Validate mode parameters
    if not req.tpot and not req.attn_bs:
        raise HTTPException(status_code=400, detail="Either tpot or attn_bs must be provided")
    
    run_id = uuid4().hex[:8]
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    args = _sanitize_args(req)
    background_tasks.add_task(_run_process, run_dir, args)
    return {"run_id": run_id, "status": "started", "log": str(run_dir / "output.log")}
```

### Step 5.4: Modify fetch_csv_results endpoint

Modify `fetch_csv_results` endpoint (lines 246-295) to support both tpot and attn_bs:

```python
@api.get("/fetch_csv_results")
def fetch_csv_results(
    device_type: str,
    model_type: str,
    tpot: Optional[int] = None,     # Retained: for constraint mode
    attn_bs: Optional[int] = None,  # New: for direct mode
    kv_len: int,
    serving_mode: str = "AFD",
    deployment_mode: str = "Homogeneous",
    device_type2: Optional[str] = None,
    micro_batch_num: Optional[int] = 1,
    total_die: Optional[int] = None,
):
    # Validate deployment_mode
    if deployment_mode not in ("Homogeneous", "Heterogeneous"):
        raise HTTPException(status_code=400, detail=f"Invalid deployment_mode: {deployment_mode}. Must be 'Homogeneous' or 'Heterogeneous'")
    # Validate device_type2 is provided when deployment_mode is Heterogeneous
    if deployment_mode == "Heterogeneous" and not device_type2:
        raise HTTPException(status_code=400, detail="device_type2 is required when deployment_mode is 'Heterogeneous'")
    # Validate mode parameters
    if tpot is None and attn_bs is None:
        raise HTTPException(status_code=400, detail="Either tpot or attn_bs must be specified")

    repo_root = Path(__file__).resolve().parents[2]
    if serving_mode == "AFD":
        if deployment_mode == "Heterogeneous":
            dir_name = repo_root / "data" / "afd" / f"mbn{micro_batch_num}" / "best" / "heterogeneous"
        else:
            dir_name = repo_root / "data" / "afd" / f"mbn{micro_batch_num}" / "best" / "homogeneous"
    elif serving_mode == "DeepEP":
        if deployment_mode == "Heterogeneous":
            dir_name = repo_root / "data" / "deepep" / "heterogeneous"
        else:
            dir_name = repo_root / "data" / "deepep" / "homogeneous"
    else:
        raise HTTPException(status_code=400, detail="Unsupported serving_mode")

    # Determine file name based on mode
    if deployment_mode == "Heterogeneous":
        if tpot is not None:
            file_name = f"{device_type}_{device_type2}-{model_type}-tpot{tpot}-kv_len{kv_len}.csv"
        else:
            file_name = f"{device_type}_{device_type2}-{model_type}-bs{attn_bs}-kv_len{kv_len}.csv"
    else:
        if tpot is not None:
            file_name = f"{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}.csv"
        else:
            file_name = f"{device_type}-{model_type}-bs{attn_bs}-kv_len{kv_len}.csv"

    path = dir_name / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"read csv error: {e}")
    if total_die is not None:
        # allow both numeric and string comparisons
        df = df[df["total_die"].astype(str) == str(total_die)]
    return df.to_dict(orient="records")
```

### Step 5.5: Modify results endpoint

Modify `list_results` endpoint (lines 135-190) to support both tpot and attn_bs:

```python
@api.get("/results")
def list_results(
    model_type: str,
    device_type: str,
    total_die: int,
    tpot: Optional[int] = None,     # Retained
    attn_bs: Optional[int] = None,  # New
    kv_len: int,
    deployment_mode: str = "Homogeneous",
    device_type2: Optional[str] = None
):
    # Validate deployment_mode
    if deployment_mode not in ("Homogeneous", "Heterogeneous"):
        raise HTTPException(status_code=400, detail=f"Invalid deployment_mode: {deployment_mode}. Must be 'Homogeneous' or 'Heterogeneous'")
    # Validate device_type2 is provided when deployment_mode is Heterogeneous
    if deployment_mode == "Heterogeneous" and not device_type2:
        raise HTTPException(status_code=400, detail="device_type2 is required when deployment_mode is 'Heterogeneous'")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"

    def existing_image_urls(paths: List[str]) -> List[str]:
        existing = []
        for relative_url in paths:
            relative_path = relative_url.removeprefix("/data/")
            if (data_root / relative_path).exists():
                existing.append(relative_url)
        return existing

    # Build image candidates based on deployment mode and calculation mode
    if deployment_mode == "Heterogeneous":
        # Heterogeneous mode
        if tpot is not None:
            throughput_candidates = [
                f"/data/images/throughput/heterogeneous/{device_type}_{device_type2}-{model_type}-mbn2-total_die{total_die}.png",
                f"/data/images/throughput/heterogeneous/{device_type}_{device_type2}-{model_type}-mbn3-total_die{total_die}.png",
                f"/data/images/throughput/heterogeneous/{device_type}_{device_type2}-{model_type}-tpot{tpot}-kv_len{kv_len}.png",
            ]
            pipeline_candidates = [
                f"/data/images/pipeline/deepep/heterogeneous/{device_type}_{device_type2}-{model_type}-tpot{tpot}-kv_len{kv_len}-deepep-mbn1-total_die{total_die}.png",
                f"/data/images/pipeline/afd/heterogeneous/{device_type}_{device_type2}-{model_type}-tpot{tpot}-kv_len{kv_len}-afd-mbn2-total_die{total_die}.png",
                f"/data/images/pipeline/afd/heterogeneous/{device_type}_{device_type2}-{model_type}-tpot{tpot}-kv_len{kv_len}-afd-mbn3-total_die{total_die}.png",
            ]
        else:
            throughput_candidates = [
                f"/data/images/throughput/heterogeneous/{device_type}_{device_type2}-{model_type}-bs{attn_bs}-kv_len{kv_len}.png",
            ]
            pipeline_candidates = [
                f"/data/images/pipeline/afd/heterogeneous/{device_type}_{device_type2}-{model_type}-bs{attn_bs}-kv_len{kv_len}-afd-mbn2-total_die{total_die}.png",
                f"/data/images/pipeline/afd/heterogeneous/{device_type}_{device_type2}-{model_type}-bs{attn_bs}-kv_len{kv_len}-afd-mbn3-total_die{total_die}.png",
            ]
    else:
        # Homogeneous mode
        if tpot is not None:
            throughput_candidates = [
                f"/data/images/throughput/homogeneous/{device_type}-{model_type}-mbn2-total_die{total_die}.png",
                f"/data/images/throughput/homogeneous/{device_type}-{model_type}-mbn3-total_die{total_die}.png",
                f"/data/images/throughput/homogeneous/{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}.png",
            ]
            pipeline_candidates = [
                f"/data/images/pipeline/deepep/homogeneous/{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}-deepep-mbn1-total_die{total_die}.png",
                f"/data/images/pipeline/afd/homogeneous/{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}-afd-mbn2-total_die{total_die}.png",
                f"/data/images/pipeline/afd/homogeneous/{device_type}-{model_type}-tpot{tpot}-kv_len{kv_len}-afd-mbn3-total_die{total_die}.png",
            ]
        else:
            throughput_candidates = [
                f"/data/images/throughput/homogeneous/{device_type}-{model_type}-bs{attn_bs}-kv_len{kv_len}.png",
            ]
            pipeline_candidates = [
                f"/data/images/pipeline/afd/homogeneous/{device_type}-{model_type}-bs{attn_bs}-kv_len{kv_len}-afd-mbn2-total_die{total_die}.png",
                f"/data/images/pipeline/afd/homogeneous/{device_type}-{model_type}-bs{attn_bs}-kv_len{kv_len}-afd-mbn3-total_die{total_die}.png",
            ]
    throughput_images = existing_image_urls(throughput_candidates)
    pipeline_images = existing_image_urls(pipeline_candidates)
    return {"throughput_images": throughput_images, "pipeline_images": pipeline_images}
```

### Step 5.6: Commit Webapp backend changes

```bash
git add webapp/backend/main.py
git commit -m "feat(webapp): add attn_bs parameter to RunRequest and update API endpoints for dual-mode support"
```

---

## Task 6: Frontend State Management (app.js)

**Files:**
- Modify: `webapp/frontend/app.js:1-256`

### Step 6.1: Update DEFAULT_CSV_SELECTION

Modify `DEFAULT_CSV_SELECTION` (lines 8-18) to use `attnBs` instead of `tpot`:

```javascript
const DEFAULT_CSV_SELECTION = {
  servingMode: "AFD",
  deploymentMode: "Heterogeneous",
  deviceType: "ASCENDDAVID120",
  deviceType2: "ASCEND910B2",
  modelType: "DEEPSEEK_V3",
  attnBs: 128,        // New: replaces tpot
  kvLen: 4096,
  microBatchNum: 3,
  totalDie: 128,
};
```

### Step 6.2: Remove tpot from LEGACY_DEFAULT_CSV_SELECTION

Remove or update the legacy selection (lines 20-23):

```javascript
const LEGACY_DEFAULT_CSV_SELECTION = {
  ...DEFAULT_CSV_SELECTION,
  deviceType: "ASCENDA3_Pod",
  // tpot removed
};
```

### Step 6.3: Update normalizeCsvSelection function

Modify `normalizeCsvSelection` function (lines 76-85) to handle `attnBs`:

```javascript
function normalizeCsvSelection(selection = {}) {
  return {
    ...DEFAULT_CSV_SELECTION,
    ...selection,
    attnBs: Number(selection.attnBs ?? DEFAULT_CSV_SELECTION.attnBs),
    kvLen: Number(selection.kvLen ?? DEFAULT_CSV_SELECTION.kvLen),
    microBatchNum: Number(selection.microBatchNum ?? DEFAULT_CSV_SELECTION.microBatchNum),
    totalDie: Number(selection.totalDie ?? DEFAULT_CSV_SELECTION.totalDie),
  };
}
```

### Step 6.4: Update deriveCsvSelectionFromRunParams function

Modify `deriveCsvSelectionFromRunParams` function (lines 94-107) to handle both modes:

```javascript
function deriveCsvSelectionFromRunParams(params = {}) {
  const hasTpot = params.tpot && params.tpot.length > 0;
  const hasAttnBs = params.attn_bs && params.attn_bs.length > 0;
  
  return {
    servingMode: params.serving_mode || DEFAULT_CSV_SELECTION.servingMode,
    deploymentMode: params.deployment_mode || DEFAULT_CSV_SELECTION.deploymentMode,
    deviceType: DEVICE_VALUE_TO_NAME[params.device_type] || params.device_type || DEFAULT_CSV_SELECTION.deviceType,
    deviceType2: DEVICE_VALUE_TO_NAME[params.device_type2] || params.device_type2 || DEFAULT_CSV_SELECTION.deviceType2,
    modelType: MODEL_VALUE_TO_NAME[params.model_type] || params.model_type || DEFAULT_CSV_SELECTION.modelType,
    attnBs: hasAttnBs ? Number(firstListValue(params.attn_bs, DEFAULT_CSV_SELECTION.attnBs)) : DEFAULT_CSV_SELECTION.attnBs,
    kvLen: Number(firstListValue(params.kv_len, DEFAULT_CSV_SELECTION.kvLen)),
    microBatchNum: Number(
      params.serving_mode === "DeepEP"
        ? 1
        : firstListValue(params.micro_batch_num, DEFAULT_CSV_SELECTION.microBatchNum),
    ),
  };
}
```

### Step 6.5: Update shouldSeedFromLatestRun function

Modify `shouldSeedFromLatestRun` function (lines 109-136) to check `attnBs` instead of `tpot`:

```javascript
function shouldSeedFromLatestRun(storedSelection, latestRun) {
  if (!latestRun?.params) {
    return false;
  }

  if (!storedSelection || Object.keys(storedSelection).length === 0) {
    return true;
  }

  const normalizedStored = normalizeCsvSelection(storedSelection);
  const matchesCurrentDefault =
    normalizedStored.servingMode === DEFAULT_CSV_SELECTION.servingMode &&
    normalizedStored.deviceType === DEFAULT_CSV_SELECTION.deviceType &&
    normalizedStored.modelType === DEFAULT_CSV_SELECTION.modelType &&
    normalizedStored.attnBs === DEFAULT_CSV_SELECTION.attnBs &&
    normalizedStored.kvLen === DEFAULT_CSV_SELECTION.kvLen &&
    normalizedStored.microBatchNum === DEFAULT_CSV_SELECTION.microBatchNum;

  const matchesLegacyDefault =
    normalizedStored.servingMode === LEGACY_DEFAULT_CSV_SELECTION.servingMode &&
    normalizedStored.deviceType === LEGACY_DEFAULT_CSV_SELECTION.deviceType &&
    normalizedStored.modelType === LEGACY_DEFAULT_CSV_SELECTION.modelType &&
    normalizedStored.attnBs === LEGACY_DEFAULT_CSV_SELECTION.attnBs &&
    normalizedStored.kvLen === LEGACY_DEFAULT_CSV_SELECTION.kvLen &&
    normalizedStored.microBatchNum === LEGACY_DEFAULT_CSV_SELECTION.microBatchNum;

  return matchesCurrentDefault || matchesLegacyDefault;
}
```

### Step 6.6: Commit app.js changes

```bash
git add webapp/frontend/app.js
git commit -m "feat(frontend): update state management to use attnBs instead of tpot"
```

---

## Task 7: RunForm.vue Changes

**Files:**
- Modify: `webapp/frontend/components/RunExperimentTab/RunForm.vue:1-423`

### Step 7.1: Add attnBsInput reactive reference

Add `attnBsInput` reference in setup function (around line 188):

```javascript
const attnBsInput = ref('128');  // New: batchsize input
const kvLenInput = ref('4096');
const mbnInput = ref('3');
const isSubmitting = ref(false);
const error = ref(null);
```

### Step 7.2: Remove tpotInput or make optional

Change `tpotInput` to optional (line 188):

```javascript
const tpotInput = ref('');  // Changed: empty default, optional
```

### Step 7.3: Modify template to add attn_bs field

Modify the "Simulation Targets" section (lines 91-105) to add attn_bs and optionally keep tpot:

```vue
<div class="section">
  <h4>Simulation Targets</h4>
  <div class="field">
    <label>Attention Batch Size (comma separated)</label>
    <input v-model="attnBsInput" placeholder="64,128,256" />
  </div>
  <div class="field">
    <label>TPOT Targets (optional, comma separated)</label>
    <input v-model="tpotInput" placeholder="20,50,70,100,150" />
    <p class="field-note">If TPOT is specified, uses constraint mode. If empty, uses direct calculation mode with attn_bs.</p>
  </div>
  <div class="field">
    <label>KV Length (comma separated)</label>
    <input v-model="kvLenInput" placeholder="2048,4096,8192" />
  </div>
  <div class="field">
    <label>Micro Batch Numbers (comma separated)</label>
    <input v-model="mbnInput" placeholder="2,3" />
  </div>
</div>
```

### Step 7.4: Add CSS for field-note

Add CSS for the new `field-note` class (around line 400):

```css
.field-note {
  font-size: 12px;
  color: #64748b;
  margin-top: 4px;
}
```

### Step 7.5: Modify handleSubmit to handle both modes

Modify `handleSubmit` function (lines 263-282) to handle both modes:

```javascript
const handleSubmit = async () => {
  const parsedTpot = parseList(tpotInput.value);
  const parsedAttnBs = parseList(attnBsInput.value);
  
  // Validation: at least one must be provided
  if (parsedTpot.length === 0 && parsedAttnBs.length === 0) {
    error.value = 'Either TPOT or Attention Batch Size must be provided';
    return;
  }

  isSubmitting.value = true;
  error.value = null;

  try {
    const payload = {
      ...form.value,
      tpot: parsedTpot.length > 0 ? parsedTpot : null,
      attn_bs: parsedAttnBs.length > 0 ? parsedAttnBs : null,
      kv_len: parseList(kvLenInput.value),
      micro_batch_num: parseList(mbnInput.value)
    };

    const result = await api.startRun(payload);
    addToHistory(result.run_id, payload);
  } catch (err) {
    error.value = err && err.message ? err.message : 'Failed to start run';
  } finally {
    isSubmitting.value = false;
  }
};
```

### Step 7.6: Update return object to include attnBsInput

Add `attnBsInput` to the return object (around line 284):

```javascript
return {
  form,
  tpotInput,
  attnBsInput,  // New
  kvLenInput,
  mbnInput,
  modelOptions: MODEL_OPTIONS,
  deviceOptions: DEVICE_OPTIONS,
  isSubmitting,
  error,
  handleSubmit,
  deviceTypeLabel,
  minDieLabel,
  maxDieLabel,
  dieStepLabel,
  deviceType2Label,
  minDie2Label,
  maxDie2Label,
  dieStep2Label
};
```

### Step 7.7: Commit RunForm.vue changes

```bash
git add webapp/frontend/components/RunExperimentTab/RunForm.vue
git commit -m "feat(frontend): add attn_bs input field to RunForm and support dual-mode submission"
```

---

## Task 8: CsvSelector.vue Changes

**Files:**
- Modify: `webapp/frontend/components/ResultsTab/CsvSelector.vue:1-365`

### Step 8.1: Update DEFAULT_SELECTION

Modify `DEFAULT_SELECTION` (lines 96-106) to use `attnBs` instead of `tpot`:

```javascript
const DEFAULT_SELECTION = {
  servingMode: 'AFD',
  deploymentMode: 'Heterogeneous',
  deviceType: 'ASCENDDAVID120',
  deviceType2: 'ASCEND910B2',
  modelType: 'DEEPSEEK_V3',
  attnBs: 128,  // New: replaces tpot
  kvLen: 4096,
  microBatchNum: 3,
  totalDie: 128
};
```

### Step 8.2: Update normalizeSelection function

Modify `normalizeSelection` function (lines 145-162) to handle `attnBs`:

```javascript
function normalizeSelection(selection, includeTotalDie = false) {
  const normalized = {
    servingMode: selection.servingMode || DEFAULT_SELECTION.servingMode,
    deploymentMode: selection.deploymentMode || DEFAULT_SELECTION.deploymentMode,
    deviceType: selection.deviceType || DEFAULT_SELECTION.deviceType,
    deviceType2: selection.deviceType2 || DEFAULT_SELECTION.deviceType2,
    modelType: selection.modelType || DEFAULT_SELECTION.modelType,
    attnBs: Number(selection.attnBs) || DEFAULT_SELECTION.attnBs,  // Changed
    kvLen: Number(selection.kvLen) || DEFAULT_SELECTION.kvLen,
    microBatchNum: Number(selection.microBatchNum) || DEFAULT_SELECTION.microBatchNum
  };

  if (includeTotalDie) {
    normalized.totalDie = Number(selection.totalDie) || DEFAULT_SELECTION.totalDie;
  }

  return normalized;
}
```

### Step 8.3: Update template to use attnBs

Modify the field for TPOT (lines 59-65) to use attnBs:

```vue
<label class="field">
  <span>Attention Batch Size</span>
  <input type="number" v-model.number="selection.attnBs" list="attnbs-list" min="1" />
  <datalist id="attnbs-list">
    <option v-for="bs in attnBsSuggestions" :key="bs" :value="bs" />
  </datalist>
</label>
```

### Step 8.4: Add ATTN_BS_SUGGESTIONS constant

Add new constant (around line 137):

```javascript
const ATTN_BS_SUGGESTIONS = [64, 128, 256, 512, 1024];
```

### Step 8.5: Update handleLoadCsv to use attnBs

Modify `handleLoadCsv` function (lines 200-227) to pass `attn_bs` instead of `tpot`:

```javascript
const handleLoadCsv = async () => {
  isLoading.value = true;
  error.value = null;

  const params = normalizeSelection(selection.value);
  try {
    const data = await api.fetchCsvResults({
      serving_mode: params.servingMode,
      deployment_mode: params.deploymentMode,
      device_type: params.deviceType,
      device_type2: params.deviceType2,
      model_type: params.modelType,
      attn_bs: String(params.attnBs),  // Changed
      kv_len: String(params.kvLen),
      micro_batch_num: String(params.microBatchNum)
    });
    setCsvSelection(params);
    emit('csv-loaded', data || []);
  } catch (err) {
    if (String(err.message).includes('404')) {
      error.value = 'CSV file not found. Run a simulation first or pick a different CSV selection.';
    } else {
      error.value = err.message;
    }
  } finally {
    isLoading.value = false;
  }
};
```

### Step 8.6: Update return object

Update return object to include `attnBsSuggestions` (around line 233):

```javascript
return {
  selection,
  isLoading,
  error,
  deviceTypeLabel,
  deviceType2Label,
  servingModes: SERVING_MODES,
  deploymentModes: DEPLOYMENT_MODES,
  modelOptions: MODEL_OPTIONS,
  deviceOptions: DEVICE_OPTIONS,
  attnBsSuggestions: ATTN_BS_SUGGESTIONS,  // Changed
  kvLenSuggestions: KV_LEN_SUGGESTIONS,
  microBatchOptions: MICRO_BATCH_OPTIONS,
  handleLoadCsv
};
```

### Step 8.7: Commit CsvSelector.vue changes

```bash
git add webapp/frontend/components/ResultsTab/CsvSelector.vue
git commit -m "feat(frontend): update CsvSelector to use attnBs instead of tpot"
```

---

## Task 9: ThroughputCharts.vue Changes

**Files:**
- Modify: `webapp/frontend/components/VisualizationsTab/ThroughputCharts.vue:1-443`

### Step 9.1: Update DEFAULT_PARAMS

Modify `DEFAULT_PARAMS` (lines 131-141) to use `attnBs`:

```javascript
const DEFAULT_PARAMS = {
  deviceType: 'ASCENDA3_Pod',
  deviceType2: 'ASCENDDAVID121',
  modelType: 'DEEPSEEK_V3',
  totalDie: 128,
  attnBs: 128,  // New: replaces tpot
  kvLen: 4096,
  servingMode: 'AFD',
  microBatchNum: 3,
  deploymentMode: 'Homogeneous'
};
```

### Step 9.2: Add ATTN_BS_SUGGESTIONS constant

Add new constant (around line 167):

```javascript
const ATTN_BS_SUGGESTIONS = [64, 128, 256, 512, 1024];
```

### Step 9.3: Update template to use attnBs

Modify the field for TPOT (lines 55-61) to use attnBs:

```vue
<label class="field">
  <span>Attention Batch Size</span>
  <input type="number" v-model.number="params.attnBs" list="attnbs-list" min="1" />
  <datalist id="attnbs-list">
    <option v-for="bs in attnBsSuggestions" :key="bs" :value="bs" />
  </datalist>
</label>
```

### Step 9.4: Update loadCharts function

Modify `loadCharts` function (lines 204-242) to use `attn_bs`:

```javascript
const loadCharts = async () => {
  loading.value = true;
  error.value = null;
  missingImages.clear();

  try {
    const apiParams = {
      device_type: params.deviceType,
      model_type: params.modelType,
      total_die: String(params.totalDie),
      attn_bs: String(params.attnBs),  // Changed
      kv_len: String(params.kvLen),
      deployment_mode: params.deploymentMode
    };
    if (params.deploymentMode === 'Heterogeneous') {
      apiParams.device_type2 = params.deviceType2;
    }
    const data = await api.getResults(apiParams);
    throughputImages.value = Array.isArray(data.throughput_images) ? data.throughput_images : [];
    pipelineImages.value = Array.isArray(data.pipeline_images) ? data.pipeline_images : [];
    setCsvSelection({
      servingMode: params.servingMode || 'AFD',
      deploymentMode: params.deploymentMode,
      deviceType: params.deviceType,
      deviceType2: params.deviceType2,
      modelType: params.modelType,
      totalDie: Number(params.totalDie),
      attnBs: Number(params.attnBs),  // Changed
      kvLen: Number(params.kvLen),
      microBatchNum: Number(params.microBatchNum) || 3
    });
  } catch (err) {
    error.value = err.message;
    throughputImages.value = [];
    pipelineImages.value = [];
  } finally {
    loading.value = false;
  }
};
```

### Step 9.5: Update return object

Update return object to include `attnBsSuggestions` (around line 252):

```javascript
return {
  params,
  throughputImages,
  pipelineImages,
  loading,
  error,
  missingImages,
  markMissing,
  loadCharts,
  deviceTypeLabel,
  deviceType2Label,
  deviceOptions: DEVICE_OPTIONS,
  modelOptions: MODEL_OPTIONS,
  deploymentModes: DEPLOYMENT_MODES,
  attnBsSuggestions: ATTN_BS_SUGGESTIONS,  // Changed
  kvLenSuggestions: KV_LEN_SUGGESTIONS
};
```

### Step 9.6: Commit ThroughputCharts.vue changes

```bash
git add webapp/frontend/components/VisualizationsTab/ThroughputCharts.vue
git commit -m "feat(frontend): update ThroughputCharts to use attnBs instead of tpot"
```

---

## Task 10: Integration Testing

**Files:**
- None (run tests)

### Step 10.1: Test direct calculation mode with CLI

Run the simulator with direct mode:

```bash
python src/cli/main.py --serving_mode AFD --model_type deepseek-ai/DeepSeek-V3 --device_type Ascend_A3Pod --attn_bs 64 128 --kv_len 4096 --min_die 16 --max_die 128 --die_step 16 --micro_batch_num 2
```

Expected: CSV files generated in `data/afd/mbn2/homogeneous/` with names like `ASCENDA3_Pod-DEEPSEEK_V3-bs64-kv_len4096.csv` and `ASCENDA3_Pod-DEEPSEEK_V3-bs128-kv_len4096.csv`

### Step 10.2: Test constraint mode with CLI

Run the simulator with constraint mode:

```bash
python src/cli/main.py --serving_mode AFD --model_type deepseek-ai/DeepSeek-V3 --device_type Ascend_A3Pod --tpot 50 --kv_len 4096 --min_die 16 --max_die 128 --die_step 16 --micro_batch_num 2
```

Expected: CSV files generated in `data/afd/mbn2/homogeneous/` with names like `ASCENDA3_Pod-DEEPSEEK_V3-tpot50-kv_len4096.csv`

### Step 10.3: Test Webapp API

Start the webapp:

```bash
python -m webapp.backend.main
```

Expected: Server starts on http://127.0.0.1:8000

### Step 10.4: Test Webapp frontend

Open browser and test the UI:
1. Navigate to Run Experiment tab
2. Enter attn_bs values (e.g., 64, 128)
3. Leave tpot empty
4. Click "Start Run"
5. Check results in Results tab

Expected: Simulation runs and CSV files are generated with batchsize naming.

### Step 10.5: Final commit

```bash
git add docs/superpowers/specs/ docs/superpowers/plans/
git commit -m "docs: add design spec and implementation plan for direct calculation mode"
```

---

## Summary

This plan implements:
1. **CLI dual-mode support** - constraint mode (TPOT) and direct mode (attn_bs)
2. **Config mode-aware initialization**
3. **Search algorithms** - direct calculation without TPOT constraint
4. **Webapp API** - dual-mode parameter support
5. **Frontend UI** - attn_bs input fields and state management

Total tasks: 10
Estimated time: 2-3 hours for skilled developer

---

*Plan written: 2026-04-24*