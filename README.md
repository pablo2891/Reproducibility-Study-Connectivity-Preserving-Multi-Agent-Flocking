# Reproducibility-Study-Connectivity-Preserving-Multi-Agent-Flocking
Reproducing the TAC 2009 paper “Hybrid Control for Connectivity Preserving Flocking” on the Robotarium Python simulator.

## Setup
- Python 3.10+ recommended. Install dependencies: `python3 -m pip install -r requirements.txt`.
- Robotarium simulator is pulled from PyPI via `robotarium-python-simulator`; no extra ROS setup required.
- If you want PDF text extraction, install `PyPDF2` or `pdfminer.six` (optional).

## How to run the paper experiment (30 agents)
- Default paper-matching run (30 agents on a ring, r=0.15 m, R=0.5 m):  
  `python3 scripts/run_paper_experiment.py --show-figure --save-plots`
- Outputs: `paper_experiment_output/metrics.npz`, `paper_experiment_output/metrics.png`.
- Useful flags:
  - `--num-agents N` (Robotarium hardware cap is 20; simulation can exceed).
  - `--initial-radius RING_RADIUS` (meters, keep ≤0.45 for workspace).
  - `--iterations K` (simulation steps).
  - `--real-time` to slow down for visualization; drop it for faster headless runs.

## How to run the 20-robot hybrid (double-integrator style)
- Run with Robotarium SI interface and hybrid topology logic:  
  `python3 scripts/robotarium_double_integrator_flocking_20.py --show-figure --hold-figure --save-plots`
- Outputs (in `robotarium_double_int_output/`):
  - `metrics.npz`: λ2 (connectivity), min distance, velocity disagreement over time.
  - `metrics.png`: plots of the above.
  - `velocities.png`: final positions + edges + velocity arrows (useful to verify alignment).
- Flags:
  - `--hold-figure` keeps the Robotarium window open at the end (requires `--show-figure`).
  - `--steps`, `--dt` override runtime; defaults are long for convergence.
  - `--seed` sets initial random velocities.

## Other scripts
- `scripts/main_robotarium_flocking.py`: lighter demo flocking run.
- `scripts/run_paper_experiment.py`: full reproduction with metrics logging.
- `scripts/robotarium_double_integrator_flocking_20.py`: 20-robot hybrid controller mapped to Robotarium SI control.
- `scripts/controllers.py`: potential-based flocking with velocity alignment.
- `scripts/topology_control.py`: hysteresis-based edge maintenance.
- `scripts/utils.py`: proximity graph, connectivity check, ring initial conditions, metrics (Fiedler value, min distance).

## What to expect
- Plots track algebraic connectivity (λ2), minimum inter-agent distance, edge count (paper run), and velocity disagreement (20-robot hybrid).
- Initial layout matches the paper: agents evenly spaced on a circle, random initial velocities (through controller state). For the 20-robot hybrid run, a velocity arrows snapshot (`velocities.png`) helps confirm flock alignment.

## Tips
- For faster batch runs: omit `--show-figure`, reduce `--iterations`, and keep `sim_in_real_time` off (default).
- Tune `alpha_potential` and `alpha_align` in `run_paper_experiment.py` if you need stronger cohesion or velocity consensus.
