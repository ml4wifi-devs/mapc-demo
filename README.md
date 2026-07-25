# JAXWiFi: An Open-Source Ecosystem for Learning-Based Multi-AP Coordination beyond Wi-Fi 8

Interactive browser demo of the [ml4wifi-devs](https://github.com/ml4wifi-devs)
open-source ecosystem for Coordinated Spatial Reuse (Co-SR) research, accompanying the
JAXWiFi demo paper.

Co-SR is an IEEE 802.11bn multi-AP coordination (MAPC) scheme in which neighboring APs
transmit simultaneously to selected stations with reduced power. This demo lets you:

- build a Wi-Fi topology by hand or pick a predefined one (TGax scenarios),
- configure the channel, PHY, and the scheduling method,
- run several methods on **exactly the same scenario** and compare them live on one chart.

## Methods available

- **H-MAB** — hierarchical multi-armed bandits, the paper's learning scheduler
- **Flat MAB** — single-level bandit ablation baseline
- **T-Optimal** — MILP upper bound on total throughput
- **F-Optimal** — MILP optimum of max-min fairness
- **T-Meta** — metaheuristic search (SA / RRHC / tabu) approximating T-Optimal
- **F-Meta** — fairness column generation approximating F-Optimal
- **FM4WiFi** — generative flow-matching scheduler with a surrogate model
- **MAPC-Surrogate** — random configurations scored by a GNN surrogate model
- **DCF** — legacy 802.11 channel access (discrete event simulation)
- **SR** — 802.11ax OBSS/PD spatial reuse (discrete event simulation)
- **Random (single TX)** — legacy-like random single-transmission baseline

Scenarios and the Monte Carlo channel simulator come from
[mapc-optimal-research](https://github.com/ml4wifi-devs/mapc-optimal-research) and
[mapc-sim](https://github.com/ml4wifi-devs/mapc-sim) (JAX-based, TGax path loss,
optional Nakagami-m fading).

---

## 1. Installation

Requirements: **Python ≥ 3.12** (3.13 recommended) and git. One command:

```bash
./install.sh
```

It creates `.venv` and installs every ecosystem package directly from GitHub —
no manual checkouts, no environment variables. `lai4wifi` and `mapc-surrogate`
are installed as editable clones (kept in `.venv/src`) so their trained model
checkpoints ship with them.

## 2. Running

```bash
./run_demo.sh          # serves http://localhost:8000
./run_demo.sh 8765     # custom port
```

First notes:
- the first simulation step of a run triggers JAX JIT compilation — a short initial
  pause is normal;
- the server keeps model checkpoints and JIT caches in memory, so subsequent FM4WiFi
  runs start much faster than the first one.

## 3. Using the demo

The screen has three panels: **Network topology** (left top), **configuration tabs**
(right), and the **effective data rate chart** (left bottom).

### 3.1 Topology panel

Two modes, shown by the badge in the panel header:

- **preview** — a predefined scenario is selected; the canvas renders it (APs as ×,
  stations as dots, association lines, walls as thick lines, meter grid). Read-only.
  Press **Customize** to copy the current topology into custom mode and edit it.
- **custom — editable** — the *Custom* scenario is selected; you draw the network:
  - **right click** — place a new AP (it becomes the selected AP and gets a new color),
  - **left click** — place a station associated with the selected AP,
  - **click an AP** — select it (dashed ring), so new stations attach to it,
  - **drag** — move any node,
  - **double click** — delete a node (deleting an AP deletes its stations),
  - **Undo** — remove the most recently placed node of the selected AP,
  - **Clear** — remove everything.

Every AP needs at least one station before you can start a run.

### 3.2 Scenario tab

- **Scenario** — the catalog: custom, small office (2×2 rooms with walls), TGax
  residential (random placement), symmetric residential (the paper's 2×4 room
  topology), TGax enterprise, random open space, spatial-reuse line, toy scenarios,
  indoor small BSSs (hex grid). Each has its own parameters (grid size, distances,
  STAs per AP, topology seed, …).
- **Simulation & channel** — applies to every scenario and every method: the shared
  **time horizon** (number of TXOPs — learning steps, search iterations; DCF/SR
  simulate the equivalent channel time, one TXOP ≈ 5.5 ms), channel width (20/40/80/160 MHz),
  TGax path-loss variant (**residential** for apartment scenarios — 5 m breaking
  point, 5 dB wall loss; **enterprise** otherwise — 10 m, 7 dB), noise σ, optional
  Nakagami-m fading, max TX power, TX power step and number of TX power levels
  available to the agents.

Every field has a **ⓘ tooltip** explaining its effect.

**Important:** any change in this tab (or any edit of a custom topology) stops all
running simulations and resets the chart. This is deliberate — all curves on the chart
always refer to one identical scenario, so comparisons are fair.

### 3.3 Method tab

Pick a method and its parameters:

- **H-MAB / Flat MAB** — choose the bandit algorithm (UCB, ε-greedy, Softmax, Exp3,
  Thompson sampling, normal Thompson sampling) and its hyperparameters. For H-MAB each
  of the three hierarchy levels (AP-set selection → station selection → TX power) has
  its own parameter block; defaults come from the tuned research configuration.
  Flat MAB is only feasible for small networks (roughly ≤ 4 APs) — the action space
  grows exponentially and the demo refuses oversized configurations.
- **T-Optimal / F-Optimal** — no parameters; the MILP solver (CBC) can take seconds to
  minutes depending on network size.
- **DCF / SR** — warmup and number of independent runs; the simulated channel time is
  derived from the time horizon (n_steps × TXOP). **Slow**: this is a per-frame discrete
  event simulation; expect minutes of compute per simulated second even on small
  topologies. Keep the horizon short for a quick estimate.
- **T-Meta** — approximate T-Optimal: pick the algorithm (simulated annealing,
  random-restart hill climbing, tabu search) plus its knobs (temperature decay, restart
  threshold, tabu list size). The search is JIT-compiled and runs as a whole, so the
  curve appears when the search finishes (a few seconds).
- **F-Meta** — approximate F-Optimal via primal-only column generation with tabu
  pricing; the chart shows the total throughput of the best fair solution, and the run
  status reports the achieved min per-station rate and Jain's index.
- **FM4WiFi** — candidates per step, top-k evaluated, surrogate vs simulator scoring;
  the number of observation-generation rounds follows from the time horizon. Trained on
  specific topology distributions; hand-drawn topologies may be out-of-distribution.
- **MAPC-Surrogate** — candidates per step, top-k evaluated, risk-averse scoring toggle;
  draws random Co-SR configurations and scores them all in one batched forward pass of
  a GNN surrogate model — no learning, no generation, no iterative search. Same
  out-of-distribution caveat as FM4WiFi.

### 3.4 Running comparisons

1. Configure the scenario once.
2. Pick a method, press **▶ Start run**. The run appears in the run list with its
   color and live status (`building → running → done`, or `error: …`).
3. Repeat with other methods (or the same method with different parameters) — each run
   gets its own color and legend entry; runs execute concurrently.
4. **■ Stop all** stops everything; each run also has its own small **stop** button.
5. **↺ Reset chart** clears all runs and colors.

Suggested first comparison (small office, defaults, 600-TXOP horizon): H-MAB (UCB),
then T-Optimal, F-Optimal, T-Meta (SA), and Random. You should see the H-MAB curve climb
from the Random level toward the T-Optimal dashed line.

### 3.5 Saving and loading results

Save downloads everything needed to reproduce or post-process the comparison as
one JSON file: the scenario and its parameters, the channel/PHY settings, and every run
with its method, its parameters, and its recorded data rates.

```jsonc
{
  "format": "jaxwifi-demo-results", "version": 1, "created": "2026-07-24T18:00:00.000Z",
  "scenario": {
    "id": "small_office",
    "params": {"d_ap": 25, "d_sta": 2},
    "custom": {"aps": []},                  // hand-drawn topology, when id == "custom"
    "globals": {"n_steps": 600, "channel_width": 20, "path_loss": "enterprise", ...}
  },
  "runs": [{
    "label": "H-MAB (UCB) #1", "method": "hmab", "method_name": "H-MAB",
    "kind": "curve",                        // "curve" or "hline"
    "state": "done",
    "params": {"agent_type": "UCB", "seed": 42, "params_lvl1": {"c": 1.5, "gamma": 0.6}, ...},
    "steps": [1, 2, 3],                     // x values (TXOP index)
    "data_rate": [98.1, 104.7, 111.2],      // y values [Mb/s]
    "hline": null, "ci": null               // one-shot methods: value + 95% CI instead
  }]
}
```

Load reads such a file back: it restores the scenario, all channel settings, the
custom topology if there was one, and preloads the method panel with the first recorded
run, so the same configuration can be run again. If the file contains runs, you are asked
whether to also draw them on the chart — loaded runs are inert (nothing is re-simulated),
they just come back with their curves and colors. The file records the exact configuration,
so a loaded setup re-runs to the same numbers. Every method is fully reproducible
as each carries an explicit seed.

### 3.6 The chart

- x: transmission opportunity (step); y: effective data rate of the whole network [Mb/s],
  anchored at 0.
- Learning/search methods draw curves; one-shot methods (T-/F-Optimal, DCF, SR) draw
  dashed horizontal lines with the value in the legend (DCF/SR also get a translucent
  95% CI band).
- **EMA slider** — exponential moving-average smoothing applied client-side; the raw
  signal stays visible as a faint line behind the smoothed one. Set to 0 to disable.
- Hover for exact values; the toolbar (top right of the chart) offers zoom/pan/PNG
  export; the app follows your system light/dark theme.

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| FM4WiFi / MAPC-Surrogate run ends with a checkpoint error | `.venv/src/lai4wifi` or `.venv/src/mapc-surrogate` missing — rerun `./install.sh`. |
| Flat MAB errors with *action space is too large* | Expected guard. Use H-MAB, fewer APs/stations, or fewer TX power levels. |
| DCF/SR runs for many minutes | Normal — detailed DES. Reduce simulated time or topology size, or stop the run. |
| First run of a method stalls a few seconds | JAX JIT compilation; subsequent steps are fast. |
| Chart resets "by itself" | A scenario or channel parameter changed (or custom topology edited) — by design. |
| `Custom topology needs at least one AP…` | Place at least one AP (right click) with one station (left click). |
| Port already in use | `./run_demo.sh 8765` or kill the stale `uvicorn` process. |

## 5. Architecture

```
mapc-demo/
├── backend/
│   ├── app.py            FastAPI app: REST + WebSocket + static file serving
│   ├── scenarios.py      scenario registry: parameter schemas (with tooltips) + builders
│   │                     on top of mapc_research.envs; custom-topology builder
│   ├── runs.py           RunManager: one worker thread per run, stop events,
│   │                     message queue → WebSocket broadcast
│   └── methods/          one runner module per method family; each registers
│       │                 {id, name, kind: curve|hline, params schema, run()}
│       ├── hmab.py       H-MAB / Flat MAB / Random (mapc-mab, reinforced-lib)
│       ├── optimal.py    T-Optimal / F-Optimal (mapc-optimal)
│       ├── dcf.py        DCF / SR (mapc-dcf, SimPy DES)
│       ├── mh.py         SA / RRHC / Tabu (mapc-mh)
│       ├── fm.py         FM4WiFi (lai4wifi; lazy checkpoint loading, cached)
│       └── surrogate.py  MAPC-Surrogate (mapc-surrogate; lazy checkpoint loading, cached)
└── frontend/             vanilla JS, no build step
    ├── index.html
    ├── style.css         light/dark via prefers-color-scheme
    ├── topology.js       SVG topology editor + scenario preview renderer
    ├── chart.js          Plotly wrapper: runs, EMA, hlines, CI bands
    ├── app.js            catalog-driven forms, run lifecycle, WebSocket client
    └── vendor/plotly.min.js
```

HTTP API (all JSON):

- `GET /api/catalog` — scenario schemas, global channel parameters, method schemas
  (the frontend renders all forms from this),
- `POST /api/scenario/preview` — scenario config → node positions, associations, walls
  (used to draw the preview),
- `POST /api/run/start` — `{method, params, scenario}` → `{run_id}`,
- `POST /api/run/stop` — `{run_id}` or `{}` for all,
- `WS /ws` — stream of `{run_id, type: points|hline|status|status_msg, …}` messages.

Simulation loop (curve methods) is the canonical MAPC RL loop: per step the agent emits
`(tx_matrix, tx_power)`, the JAX simulator returns `(throughput, reward)`, the reward
feeds the next `agent.sample(reward)`.

## 6. Ecosystem repositories

| Repository | Purpose |
|---|---|
| [mapc-sim](https://github.com/ml4wifi-devs/mapc-sim) | JAX Monte Carlo simulator of Co-SR transmission opportunities (TGax path loss, Nakagami-m fading) |
| [mapc-optimal](https://github.com/ml4wifi-devs/mapc-optimal) | MILP upper bounds via column generation (T-Optimal / F-Optimal) |
| [mapc-mab](https://github.com/ml4wifi-devs/mapc-mab) | Hierarchical multi-armed bandit scheduler (H-MAB) |
| [mapc-dcf](https://github.com/ml4wifi-devs/mapc-dcf) | Discrete event simulator of DCF and 802.11ax OBSS/PD spatial reuse |
| [mapc-mh](https://github.com/ml4wifi-devs/mapc-mh) | Metaheuristic schedulers: T-Meta (SA / RRHC / tabu) and F-Meta (fairness column generation) |
| [fm4wifi](https://github.com/ml4wifi-devs/fm4wifi) | FM4WiFi generative pipeline (GNN autoencoder + flow matching + surrogate) with trained checkpoints |
| [mapc-surrogate](https://github.com/ml4wifi-devs/mapc-surrogate) | Surrogate-only scheduler: random candidates scored by a GNN+MDN surrogate model, trained checkpoint included |
| [mapc-optimal-research](https://github.com/ml4wifi-devs/mapc-optimal-research) | Common scenario abstraction and TGax topology catalog |
| [reinforced-lib](https://github.com/m-wojnar/reinforced-lib) | Reinforcement learning library providing the bandit algorithms |

## 7. Citation

If you use this demo or the ecosystem in your research, please cite:

```bibtex
@misc{wojnar2026jaxwifi,
  title  = {{JAXWiFi}: An Open-Source Ecosystem for Learning-Based Multi-AP Coordination beyond {Wi-Fi} 8},
  author = {Wojnar, Maksymilian and Rusek, Krzysztof and Kosek-Szott, Katarzyna and Szott, Szymon},
  year   = {2026},
}
```
