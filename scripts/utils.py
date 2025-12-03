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
