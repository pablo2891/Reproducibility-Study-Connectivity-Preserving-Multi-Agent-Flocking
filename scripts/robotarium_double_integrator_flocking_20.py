"""
Robotarium-based simulation of the hybrid connectivity-preserving flocking
controller (Zavlanos et al., TAC 2009) for 20 double-integrator agents.

Key features:
- Agents: double-integrator in 2D (x_dot = v, v_dot = u) with SI->unicycle mapping for Robotarium.
- Initial positions: 20 agents evenly spaced on a unit circle (radius=1.0).
- Initial velocities: uniform random in [-1, 1] per component.
- Communication: proximity graph with hysteresis bands (R=0.5, r=0.15, eps=0.05).
- Topology control: additions immediate; deletions via a max-consensus auction to keep connectivity.
- Control: velocity consensus + artificial potential gradient; acceleration clipping.
- Metrics: Fiedler eigenvalue, minimum distance, velocity disagreement (logged each step).

Run:
    python3 scripts/robotarium_double_integrator_flocking_20.py --show-figure --save-plots
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from rps.robotarium import Robotarium
from rps.utilities.transformations import create_si_to_uni_mapping, create_si_to_uni_dynamics_with_backwards_motion
from rps.utilities.barrier_certificates import create_single_integrator_barrier_certificate_with_boundary


@dataclass
class Params:
    r: float = 0.15
    R: float = 0.5
    eps: float = 0.05
    alpha_align: float = 1.5
    alpha_potential: float = 0.8
    max_acc: float = 1.2
    target_speed: float = 0.10
    speed_gain: float = 0.5
    dt: float = 0.03
    steps: int = 5000000
    comm_rounds: int = 6


def ring_initial_conditions(num_agents: int, radius: float = 0.35, seed: int | None = None):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, num_agents, endpoint=False)
    x = np.vstack((radius * np.cos(angles), radius * np.sin(angles)))
    thetas = np.zeros((1, num_agents))
    v = rng.uniform(low=-0.4, high=0.4, size=(2, num_agents))
    return np.vstack((x, thetas)), v


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, :, None] - x[:, None, :]
    return np.linalg.norm(diff, axis=0)


def is_connected(A: np.ndarray) -> bool:
    N = A.shape[0]
    visited = [False] * N
    stack = [0]
    visited[0] = True
    while stack:
        i = stack.pop()
        neighbors = np.where(A[i] != 0)[0]
        for j in neighbors:
            if not visited[j]:
                visited[j] = True
                stack.append(j)
    return all(visited)


def build_hysteresis_graph(x: np.ndarray, A_prev: np.ndarray, params: Params) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    N = x.shape[1]
    A = A_prev.copy()
    add_candidates = []
    del_candidates = []
    dists = pairwise_distances(x)
    R_add = params.R - params.eps
    R_del = params.R + params.eps
    for i in range(N):
        for j in range(i + 1, N):
            d = dists[i, j]
            if d < R_add:
                if A[i, j] == 0:
                    add_candidates.append((i, j))
                A[i, j] = A[j, i] = 1
            elif d > R_del:
                if A[i, j] == 1:
                    del_candidates.append((i, j))
            # inside hysteresis band: keep as is
    return A, add_candidates, del_candidates


def safe_deletions(A_est: np.ndarray, del_candidates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    safe = []
    for (i, j) in del_candidates:
        if A_est[i, j] == 0:
            continue
        A_tmp = A_est.copy()
        A_tmp[i, j] = A_tmp[j, i] = 0
        if is_connected(A_tmp):
            safe.append((i, j))
    return safe


def auction_delete(A_estimates: List[np.ndarray], del_candidates: List[Tuple[int, int]], dists: np.ndarray, params: Params) -> List[np.ndarray]:
    num_agents = len(A_estimates)
    R_del = params.R + params.eps
    bids = []
    for agent in range(num_agents):
        safe = safe_deletions(A_estimates[agent], del_candidates)
        if not safe:
            bids.append((0.0, -1, -1))
            continue
        best_edge = max(safe, key=lambda e: dists[e[0], e[1]])
        bid_value = max(0.0, dists[best_edge[0], best_edge[1]] - R_del)
        bids.append((bid_value, best_edge[0], best_edge[1]))

    A_current = A_estimates[0]
    for _ in range(params.comm_rounds):
        new_bids = bids.copy()
        for i in range(num_agents):
            best = bids[i]
            neighbors = np.where(A_current[i] != 0)[0]
            for j in neighbors:
                if bids[j][0] > best[0] or (math.isclose(bids[j][0], best[0]) and (bids[j][1], bids[j][2]) < (best[1], best[2])):
                    best = bids[j]
            new_bids[i] = best
        bids = new_bids

    winner_bid, winner_i, winner_j = bids[0]
    if winner_bid <= 0.0 or winner_i < 0:
        return A_estimates

    A_new = []
    for A in A_estimates:
        A_del = A.copy()
        A_del[winner_i, winner_j] = A_del[winner_j, winner_i] = 0
        A_new.append(A_del)
    return A_new


def laplacian(A: np.ndarray) -> np.ndarray:
    deg = np.diag(A.sum(axis=1))
    return deg - A


def fiedler_value(A: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(laplacian(A))
    vals = np.sort(np.real(vals))
    return float(vals[1]) if vals.size > 1 else 0.0


def velocity_disagreement(v: np.ndarray) -> float:
    v_avg = np.mean(v, axis=1, keepdims=True)
    diff = v - v_avg
    return float(np.sum(diff ** 2))


def plot_velocities(x_si_snap: np.ndarray, v_si_snap: np.ndarray, A_snap: np.ndarray, outpath: Path):
    """Save a snapshot with positions, edges, and velocity arrows."""
    plt.figure(figsize=(6, 6))
    plt.scatter(x_si_snap[0], x_si_snap[1], c="tab:blue")
    for i in range(A_snap.shape[0]):
        for j in range(i + 1, A_snap.shape[0]):
            if A_snap[i, j]:
                plt.plot([x_si_snap[0, i], x_si_snap[0, j]], [x_si_snap[1, i], x_si_snap[1, j]], "k-", alpha=0.4)
    plt.quiver(
        x_si_snap[0],
        x_si_snap[1],
        v_si_snap[0],
        v_si_snap[1],
        color="tab:red",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
    )
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.title("Positions and velocities")
    plt.savefig(outpath, dpi=200)
    plt.close()


def potential_force(xi: np.ndarray, xj: np.ndarray, params: Params) -> np.ndarray:
    diff = xi - xj
    d = np.linalg.norm(diff)
    if d < 1e-6:
        return np.zeros_like(diff)
    direction = diff / d
    if d < params.r:
        mag = (1.0 / d - 1.0 / params.r)
    elif d < params.R:
        mag = (1.0 / params.R - 1.0 / d)
    else:
        mag = 0.0
    return params.alpha_potential * mag * direction


def control_inputs(x: np.ndarray, v: np.ndarray, A: np.ndarray, params: Params) -> np.ndarray:
    N = x.shape[1]
    u = np.zeros_like(x)
    # Soft cohesion toward local neighbor centroid to keep the group tight
    for i in range(N):
        neighbors = np.where(A[i] != 0)[0]
        if neighbors.size > 0:
            centroid = np.mean(x[:, neighbors], axis=1)
            u[:, i] += 0.2 * (centroid - x[:, i])

    for i in range(N):
        force = np.zeros(2)
        align = np.zeros(2)
        neighbors = np.where(A[i] != 0)[0]
        for j in neighbors:
            force += potential_force(x[:, i], x[:, j], params)
            align += params.alpha_align * (v[:, j] - v[:, i])
        # Speed damping toward target magnitude to avoid scattering/over-speed
        speed_error = params.target_speed * (v[:, i] / (np.linalg.norm(v[:, i]) + 1e-6)) - v[:, i]
        u[:, i] = -force + align + params.speed_gain * speed_error
    # clip accelerations
    acc_norms = np.linalg.norm(u, axis=0)
    mask = acc_norms > params.max_acc
    u[:, mask] = (params.max_acc / acc_norms[mask]) * u[:, mask]
    return u


def parse_args():
    parser = argparse.ArgumentParser(description="Robotarium double-integrator flocking (20 agents, TAC 2009 style).")
    parser.add_argument("--steps", type=int, default=5000000, help="Simulation steps.")
    parser.add_argument("--dt", type=float, default=0.03, help="Timestep (s).")
    parser.add_argument("--show-figure", action="store_true", help="Render Robotarium figure.")
    parser.add_argument("--hold-figure", action="store_true", help="Keep figure open at end (requires --show-figure).")
    parser.add_argument("--real-time", action="store_true", help="Run in real time.")
    parser.add_argument("--save-plots", action="store_true", help="Save metrics plots.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for initial velocities.")
    parser.add_argument("--output-dir", type=Path, default=Path("robotarium_double_int_output"), help="Where to save metrics/plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    params = Params(dt=args.dt, steps=args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    initial_conditions, v_si = ring_initial_conditions(20, radius=0.4, seed=args.seed)

    # Robotarium setup
    r = Robotarium(
        number_of_robots=20,
        show_figure=args.show_figure,
        initial_conditions=initial_conditions,
        sim_in_real_time=args.real_time,
    )
    _, uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(
        barrier_gain=300, safety_radius=0.12
    )

    A = None
    A_estimates = None
    lambda2_log = []
    mindist_log = []
    vel_dis_log = []

    for _ in range(params.steps):
        x = r.get_poses()
        x_si = uni_to_si_states(x)

        dists = pairwise_distances(x_si)
        if A is None:
            A = (dists < params.R).astype(int)
            np.fill_diagonal(A, 0)
            if not is_connected(A):
                raise RuntimeError("Initial graph is not connected; adjust initial radius or R.")
            A_estimates = [A.copy() for _ in range(20)]
            add_cand, del_cand = [], []
        else:
            A, add_cand, del_cand = build_hysteresis_graph(x_si, A, params)

            for i, j in add_cand:
                for A_est in A_estimates:
                    A_est[i, j] = A_est[j, i] = 1

            A_estimates = auction_delete(A_estimates, del_cand, dists, params)
            A = A_estimates[0]

        # Virtual acceleration (paper law) then integrate to desired velocity
        u = control_inputs(x_si, v_si, A, params)
        v_si = v_si + params.dt * u

        # Cap desired speed before safety to reduce clipping and keep flock together
        v_norms = np.linalg.norm(v_si, axis=0)
        mask = v_norms > params.target_speed
        if np.any(mask):
            v_si[:, mask] = (params.target_speed / v_norms[mask]) * v_si[:, mask]

        # Apply barrier cert for safety
        dxi = si_barrier_cert(v_si, x_si)
        v_si = dxi.copy()  # keep internal state close to commanded safe velocity
        dxu = si_to_uni_dyn(dxi, x)

        r.set_velocities(np.arange(20), dxu)
        r.step()

        lambda2_log.append(fiedler_value(A))
        d_safe = dists + np.eye(20) * 1e6
        mindist_log.append(float(np.min(d_safe)))
        vel_dis_log.append(velocity_disagreement(v_si))

    r.call_at_scripts_end()

    lambda2_log = np.array(lambda2_log)
    mindist_log = np.array(mindist_log)
    vel_dis_log = np.array(vel_dis_log)

    np.savez(
        args.output_dir / "metrics.npz",
        lambda2=lambda2_log,
        mindist=mindist_log,
        vel_dis=vel_dis_log,
        params=params,
    )

    # Always emit velocity arrows snapshot at final time
    plot_velocities(x_si, v_si, A, args.output_dir / "velocities.png")

    if args.save_plots:
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        axes[0].plot(lambda2_log)
        axes[0].set_ylabel(r"$\lambda_2$")
        axes[0].grid(True, linestyle="--", linewidth=0.5)

        axes[1].plot(mindist_log)
        axes[1].axhline(params.r, color="r", linestyle="--", label="r")
        axes[1].set_ylabel("Min distance")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", linewidth=0.5)

        axes[2].plot(vel_dis_log)
        axes[2].set_ylabel("Velocity disagreement")
        axes[2].set_xlabel("Step")
        axes[2].grid(True, linestyle="--", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(args.output_dir / "metrics.png", dpi=200)
        plt.close(fig)

    if args.show_figure and args.hold_figure:
        # Keep the Robotarium figure open for inspection
        plt.show(block=True)


if __name__ == "__main__":
    main()
