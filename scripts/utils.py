import numpy as np


def build_proximity_graph(x_si, R):
    """
    Build adjacency matrix for proximity graph:
    A[i, j] = 1 if distance between i and j is < R and i != j.
    """
    _, N = x_si.shape
    A = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            diff = x_si[:, i] - x_si[:, j]
            dist = np.linalg.norm(diff)
            if dist < R:
                A[i, j] = 1
                A[j, i] = 1

    return A


def is_connected(A):
    """
    Simple BFS-based connectivity check.
    Returns True if the undirected graph with adjacency A is connected.
    """
    N = A.shape[0]
    visited = [False] * N

    # Start from node 0
    stack = [0]
    visited[0] = True

    while stack:
        i = stack.pop()
        neighbors = np.where(A[i, :] != 0)[0]
        for j in neighbors:
            if not visited[j]:
                visited[j] = True
                stack.append(j)

    return all(visited)


def ring_initial_conditions(num_agents, radius=0.4):
    """
    Place agents evenly on a circle of given radius (in meters).
    Returns 3 x N array [x; y; theta] for Robotarium initialization.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_agents, endpoint=False)
    xy = radius * np.vstack((np.cos(angles), np.sin(angles)))
    thetas = np.zeros((1, num_agents))
    return np.vstack((xy, thetas))


def fiedler_value(A):
    """
    Compute algebraic connectivity (second smallest eigenvalue) of Laplacian.
    """
    n = A.shape[0]
    if n < 2:
        return 0.0
    deg = np.diag(A.sum(axis=1))
    lap = deg - A
    vals = np.linalg.eigvalsh(lap)
    vals = np.sort(np.real(vals))
    return float(vals[1]) if vals.size > 1 else 0.0


def min_distance(x_si):
    """
    Minimum pairwise distance in a set of 2 x N positions.
    """
    _, n = x_si.shape
    if n < 2:
        return 0.0
    min_d = np.inf
    for i in range(n):
        diffs = x_si[:, i].reshape((2, 1)) - x_si
        dists = np.linalg.norm(diffs, axis=0)
        dists[i] = np.inf
        local_min = np.min(dists)
        if local_min < min_d:
            min_d = local_min
    return float(min_d)
