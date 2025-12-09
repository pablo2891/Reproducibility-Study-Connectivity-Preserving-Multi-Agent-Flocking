import numpy as np


def flocking_control(x_si, A, params, vel_history=None):
    """
    Simple flocking-like controller using distance-based potentials.

    x_si: 2 x N single-integrator positions
    A:    N x N adjacency matrix (0/1)
    params: dict with keys:
        - r: collision avoidance radius
        - R: communication radius
        - alpha_potential: gain on potential term
        - alpha_align: gain on velocity alignment
        - max_si_speed: max single-integrator speed
    vel_history: 2 x N array of previous single-integrator velocities (optional)
    """
    r = params["r"]
    R = params["R"]
    alpha_potential = params["alpha_potential"]
    alpha_align = params.get("alpha_align", 0.0)
    max_si_speed = params["max_si_speed"]

    _, N = x_si.shape
    dxi = np.zeros((2, N))

    # Potential-based interaction for each agent
    for i in range(N):
        xi = x_si[:, i].reshape((2, 1))
        force = np.zeros((2, 1))
        align_term = np.zeros((2, 1))

        for j in range(N):
            if i == j or A[i, j] == 0:
                continue

            xj = x_si[:, j].reshape((2, 1))
            diff = xi - xj
            dist = np.linalg.norm(diff)

            if dist < 1e-6:
                continue

            direction = diff / dist

            # Piecewise potential:
            # - strong repulsion if d < r
            # - mild attraction if r <= d <= R
            # - no interaction if d > R
            if dist < r:
                # Repulsive term ~ (1/dist - 1/r)
                mag = (1.0 / dist - 1.0 / r)
            elif dist < R:
                # Attractive term ~ (1.0 / R - 1.0 / dist)
                mag = (1.0 / R - 1.0 / dist)
            else:
                mag = 0.0

            force += alpha_potential * mag * direction

            if vel_history is not None and alpha_align > 0.0:
                align_term += alpha_align * (
                    vel_history[:, j].reshape((2, 1)) - vel_history[:, i].reshape((2, 1))
                )

        # Negative gradient direction
        dxi[:, i] = (-force + align_term).flatten()

    # Limit max speed per agent
    speeds = np.linalg.norm(dxi, axis=0)
    for i in range(N):
        if speeds[i] > max_si_speed and speeds[i] > 1e-6:
            dxi[:, i] = (max_si_speed / speeds[i]) * dxi[:, i]

    return dxi
