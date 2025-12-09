# Reproducibility-Study-Connectivity-Preserving-Multi-Agent-Flocking
Reproducing the TAC 2009 paper “Hybrid Control for Connectivity Preserving Flocking” on the Robotarium Python simulator.

## Setup
- Python 3.10+ recommended. Install dependencies: `python3 -m pip install -r requirements.txt`.
- Robotarium simulator is pulled from PyPI via `robotarium-python-simulator`; no extra ROS setup required.
- If you want PDF text extraction, install `PyPDF2` or `pdfminer.six` (optional).

## How to run the paper experiment
- Default paper-matching run (30 agents on a ring, r=0.15 m, R=0.5 m):
  `python3 scripts/run_paper_experiment.py --show-figure --save-plots`
- Outputs: metrics/plots saved to `paper_experiment_output/` (`metrics.npz`, `metrics.png`).
- Useful flags:
  - `--num-agents N` (Robotarium hardware cap is 20; simulation can exceed).
  - `--initial-radius RING_RADIUS` (meters, keep ≤0.45 for workspace).
  - `--iterations K` (simulation steps).
  - `--real-time` to slow down for visualization; drop it for faster headless runs.

## Other scripts
- `scripts/main_robotarium_flocking.py`: lighter demo flocking run.
- `scripts/run_paper_experiment.py`: full reproduction with metrics logging.
- `scripts/controllers.py`: potential-based flocking with velocity alignment.
- `scripts/topology_control.py`: hysteresis-based edge maintenance.
- `scripts/utils.py`: proximity graph, connectivity check, ring initial conditions, metrics (Fiedler value, min distance).

## What to expect
- Plots track algebraic connectivity (λ2), minimum inter-agent distance, and edge count over time—mirroring the paper’s reported metrics.
- Initial layout matches the paper: agents evenly spaced on a circle, random initial velocities (through controller state).

## Tips
- For faster batch runs: omit `--show-figure`, reduce `--iterations`, and keep `sim_in_real_time` off (default).
- Tune `alpha_potential` and `alpha_align` in `run_paper_experiment.py` if you need stronger cohesion or velocity consensus.
