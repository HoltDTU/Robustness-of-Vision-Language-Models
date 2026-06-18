# Robustness of Vision Language Models at Reading Graphs

Course project for **02445 – Statistical Evaluation for Artificial Intelligence (DTU)**.

We study how robustly two Vision Language Models (VLMs) read properties from graph images as the
**visual complexity** (number of nodes/edges) increases. Robustness is operationalized not as the
absolute performance level, but as the **slope of error degradation**: how quickly error grows per
additional node. A model can have low accuracy yet be robust (flat slope) — and vice versa.

## Research question

> As graph complexity increases, how robustly do Vision Language Models read off
> objective graph properties, where robustness is defined as the rate at which
> accuracy degrades with complexity, rather than the absolute level of accuracy?

## Study design

The study consists of **two independent sub-experiments**, each with its own graph set. The two sets
are *not* numerically comparable and are reported as standalone sub-studies.

| | Experiment 1 (monochrome) | Experiment 2 (color) |
|---|---|---|
| Tasks | Count nodes, count edges, classify directed/undirected | Count blue nodes, count red nodes |
| Complexity | 24 levels (1–24 nodes) | 24 levels (1–24 nodes) |
| Graphs | 10 distinct graphs per level = 240 | same structure (cut at ~200 predictions/model) |
| Manifest | `pilot_manifest.csv` | `correct_graph_answers.csv` |
| Predictions | `predictions.csv` | `blue_red_predictions.csv` |

**Models** (both accessed via DTU CampusAI):

- `google/gemma-4-26b-a4b`
- `alibaba/qwen-3.6-35b-a3b`

The models are **not** compared as the primary goal — they enter as a blocking/control factor so that
level differences between models do not contaminate the complexity effect. All calls are run
deterministically (`temperature=0.0`), and Qwen's "thinking mode" is disabled
(`enable_thinking=False`) to prevent the token budget from running out on complex graphs.

## Repository structure

```
.
├── generate_pilot.py          # Generates monochrome graphs + pilot_manifest.csv
├── generate_pilot_farve.py    # Generates color graphs (independent color RNG) + manifest
├── run_study.py               # Main experiment: queries the VLMs via CampusAI
├── run_study_br.py            # Color probe (blue/red nodes)
├── analyse.py                 # Statistical analysis + figures
├── MAEvsComp_plot.py          # MAE vs. complexity (standalone plot for the report)
│
├── pilot_manifest.csv         # Ground truth, experiment 1
├── correct_graph_answers.csv  # Ground truth, experiment 2 (color)
├── predictions.csv            # Model outputs, experiment 1
├── blue_red_predictions.csv   # Model outputs, experiment 2
├── png_grafer/                # Graph images (PNG)
│
├── API.env                    # CAMPUSAI_API_KEY (do NOT commit)
└── statistik_metoder.tex      # Methods section (LaTeX, Danish)
```

## Setup

```bash
pip install openai python-dotenv pandas numpy scipy statsmodels matplotlib networkx
```

Create an `API.env` file in the repo root:

```
CAMPUSAI_API_KEY=your_key_here
```

> **Note:** The CampusAI endpoint (`api.campusai.compute.dtu.dk`) is only reachable from DTU's network
> (campus or DTU VPN). On the HPC cluster, the login node is used for API calls, since the compute
> nodes are network-isolated from CampusAI.

## Running the pipeline

```bash
# 1) Generate graphs and ground-truth manifests
python generate_pilot.py          # experiment 1 (monochrome)
python generate_pilot_farve.py    # experiment 2 (color)

# 2) Run the models (requires DTU network + API.env)
python run_study.py               # experiment 1
python run_study_br.py            # experiment 2

# 3) Analyze and produce figures
python analyse.py
python MAEvsComp_plot.py
```

`run_study.py` is built to be robust against connection loss: a row is written only once *all* answers
for a graph have been received, and incomplete rows from a prior outage are cleaned up at startup. The
script can therefore be restarted and resumes where it left off.

## Statistical methods

Robustness = slope, not level. The core is a regression of error on complexity (1 degree of freedom),
which is substantially more powerful than a 24-group omnibus ANOVA.

- **OLS regression** of absolute error on node count, with **cluster-robust standard errors**
  (clustered by `image_id`); model as a blocking factor.
- **Spearman rank correlation** as a nonparametric robustness check.
- **Interaction model** (complexity × task) to test whether edges degrade faster than nodes.
- **McNemar** and **Wilcoxon signed-rank** tests for paired node-vs-edge comparisons.
- **Clopper-Pearson exact binomial intervals** for directionality accuracy.
- Logistic regression for directionality is omitted due to **perfect separation** (accuracy at the
  ceiling), where a slope cannot be estimated meaningfully.

## Key findings

**Experiment 1**

- Node counting degrades at β₁ ≈ 0.105 MAE/node; edge counting at β₁ ≈ 0.173 MAE/node.
- Edges degrade significantly faster than nodes (interaction β₃ ≈ 0.069, p ≈ 1.4·10⁻¹⁴).
- Directionality reading sits at the ceiling across the whole range (2–24 nodes) — no measurable
  degradation.

**Experiment 2**

- Qwen outperforms Gemma (combined accuracy ~62.7% vs. ~49.5%).
- Gemma degrades ~65% faster per additional node.
- Both models are near-perfect up to ~7 nodes, with sharp degradation around 10–11 nodes.

## Reproducibility

Each graph is generated from an explicit `seed` (and a separate `color_seed` in the color variant),
stored in the manifest files. The entire graph set can therefore be regenerated bit-for-bit.
