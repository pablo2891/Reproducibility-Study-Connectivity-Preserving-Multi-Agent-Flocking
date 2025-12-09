"""
Robotarium reproduction of the connectivity-preserving flocking experiment
from "Hybrid Control for Connectivity Preserving Flocking" (TAC 2009).

Default run uses 10 agents for safer/smoother Robotarium simulation.
Use --num-agents 30 to match the paper. Positions are scaled to the Robotarium
workspace (radius <= 0.45 m).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rps.robotarium import Robotarium
from rps.utilities.transformations import create_si_to_uni_mapping, create_si_to_uni_dynamics_with_backwards_motion
from rps.utilities.barrier_certificates import create_single_integrator_barrier_certificate_with_boundary

from controllers import flocking_control
from topology_control import update_graph
from utils import build_proximity_graph, ring_initial_conditions, fiedler_value, min_distance


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce connectivity-preserving flocking (TAC'09) on Robotarium.")
    parser.add_argument("--num-agents", type=int, default=10, help="Number of robots (paper uses 30; Robotarium hardware limit is 20).")
    parser.add_argument("--iterations", type=int, default=800, help="Simulation steps to run.")
    parser.add_argument("--initial-radius", type=float, default=0.4, help="Ring radius in meters (keep <=0.45 to stay inside Robotarium bounds).")
    parser.add_argument("--real-time", action="store_true", help="Run simulation in real time (slower but visually smooth).")
    parser.add_argument("--show-figure", action="store_true", help="Render the Robotarium figure.")
    parser.add_argument("--save-plots", action="store_true", help="Save metrics plot to disk.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_experiment_output"), help="Directory to store metrics.")
    return parser.parse_args()


def run():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    N = args.num_agents
    params = {
        "r": 0.15,
        "R": 0.5,
        # Gains/speed tuned down to reduce actuator saturation on Robotarium
        "alpha_potential": 0.6,
        "alpha_align": 0.3,
        "max_si_speed": 0.10,
    }

    initial_conditions = ring_initial_conditions(N, radius=args.initial_radius)

    r = Robotarium(
        number_of_robots=N,
        show_figure=args.show_figure,
        initial_conditions=initial_conditions,
        sim_in_real_time=args.real_time,
    )

    # Single-integrator mappings and barrier certificates (Robotarium utilities)
    _, uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

    prev_dxi = np.zeros((2, N))
    lambda2_log = []
    min_dist_log = []
    edge_count_log = []

    for _ in range(args.iterations):
        x = r.get_poses()           # 3 x N
        x_si = uni_to_si_states(x)  # 2 x N

        A = build_proximity_graph(x_si, params["R"])
        A = update_graph(A, x_si, params)

        dxi = flocking_control(x_si, A, params, vel_history=prev_dxi)
        dxi = si_barrier_cert(dxi, x_si)
        dxu = si_to_uni_dyn(dxi, x)

        r.set_velocities(np.arange(N), dxu)
        r.step()

        lambda2_log.append(fiedler_value(A))
        min_dist_log.append(min_distance(x_si))
        edge_count_log.append(int(np.sum(A) // 2))
        prev_dxi = dxi.copy()

    r.call_at_scripts_end()

    lambda2_log = np.array(lambda2_log)
    min_dist_log = np.array(min_dist_log)
    edge_count_log = np.array(edge_count_log)

    np.savez(
        args.output_dir / "metrics.npz",
        lambda2=lambda2_log,
        min_distance=min_dist_log,
        edge_count=edge_count_log,
        params=params,
        num_agents=N,
    )

    if args.save_plots:
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        axes[0].plot(lambda2_log)
        axes[0].set_ylabel(r"$\lambda_2$ (algebraic connectivity)")
        axes[0].grid(True, linestyle="--", linewidth=0.5)

        axes[1].plot(min_dist_log)
        axes[1].axhline(params["r"], color="r", linestyle="--", label="r (avoid)")
        axes[1].set_ylabel("Min distance [m]")
        axes[1].grid(True, linestyle="--", linewidth=0.5)
        axes[1].legend()

        axes[2].plot(edge_count_log)
        axes[2].set_ylabel("Edge count")
        axes[2].set_xlabel("Iteration")
        axes[2].grid(True, linestyle="--", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(args.output_dir / "metrics.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    run()
