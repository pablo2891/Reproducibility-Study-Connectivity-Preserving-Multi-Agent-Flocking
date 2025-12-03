import numpy as np

from utils import is_connected


def update_graph(A, x_si, params):
    """
    Simplified topology control with hysteresis:

    - Adds edges when distance < R_add
    - Deletes edges when distance > R_del, but only if connectivity is preserved

    This mimics connectivity-preserving behavior without the full auction
    mechanism from the paper.
    """
    R = params["R"]
    delta = 0.05  # hysteresis width

    R_add = R - delta   # threshold to add edges
    R_del = R + delta   # threshold to remove edges

    N = A.shape[0]
    x = x_si

    # Recompute distances and update edges
    for i in range(N):
        for j in range(i + 1, N):
            diff = x[:, i] - x[:, j]
            dist = np.linalg.norm(diff)

            if dist < R_add:
                # Ensure edge exists
                A[i, j] = 1
                A[j, i] = 1
            elif dist > R_del:
                # Candidate for deletion: only delete if graph stays connected
                if A[i, j] == 1:
                    A_temp = A.copy()
                    A_temp[i, j] = 0
                    A_temp[j, i] = 0
                    if is_connected(A_temp):
                        A[i, j] = 0
                        A[j, i] = 0
                    # else: keep edge to preserve connectivity
            # Else: keep previous status inside hysteresis band

    return A
