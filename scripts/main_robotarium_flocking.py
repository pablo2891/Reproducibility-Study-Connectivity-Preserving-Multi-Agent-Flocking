from rps.robotarium import Robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
from rps.utilities.misc import *

import numpy as np

from controllers import flocking_control
from topology_control import update_graph
from utils import build_proximity_graph


def main():
    # Number of robots (keep <= 20 if you plan to run on real Robotarium)
    N = 10

    # Parameters roughly inspired by the paper
    params = {
        "r": 0.15,        # collision avoidance radius (m)
        "R": 0.5,         # communication radius (m)
        "alpha_align": 1.0,
        "alpha_potential": 1.0,
        "max_si_speed": 0.15
    }

    # Initial conditions: Robotarium helper (random spread)
    initial_conditions = generate_initial_conditions(N)

    # Instantiate Robotarium
    r = Robotarium(
        number_of_robots=N,
        show_figure=True,
        initial_conditions=initial_conditions,
        sim_in_real_time=True
    )

    # Single-integrator <-> unicycle mappings
    _, uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

    # Barrier certificates for safety
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

    # Simulation length (iterations)
    iterations = 2000

    for _ in range(iterations):
        # Get current states
        x = r.get_poses()           # 3 x N
        x_si = uni_to_si_states(x)  # 2 x N

        # Build proximity graph from current positions
        A = build_proximity_graph(x_si, params["R"])

        # Update graph with simple topology control / hysteresis
        A = update_graph(A, x_si, params)

        # Compute flocking control in single-integrator space
        dxi = flocking_control(x_si, A, params)

        # Apply barrier certificates (physical collision safety)
        dxi = si_barrier_cert(dxi, x_si)

        # Map SI velocities to unicycle
        dxu = si_to_uni_dyn(dxi, x)

        # Send commands
        r.set_velocities(np.arange(N), dxu)
        r.step()

    # Required for Robotarium
    r.call_at_scripts_end()


if __name__ == "__main__":
    main()
