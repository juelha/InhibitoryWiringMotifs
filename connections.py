import numpy as np

def connect_one_to_one(n_pre, n_post):
    """
    1 on the main diagonal (i -> i), 0 elsewhere.
    Extra rows/cols stay 0 if sizes differ.
    """
    A = np.zeros((n_pre, n_post), dtype=np.float32)
    np.fill_diagonal(A, 1)
    return A


def connect_all_to_all(n_pre, n_post, exclude_self=True):
    """
    All ones. If square and exclude_self=True, zero the diagonal.
    """
    A = np.ones((n_pre, n_post), dtype=np.float32)
    if exclude_self and n_pre == n_post:
        np.fill_diagonal(A, 0)
    return A

def connect_random(n_pre, n_post, p=0.1, exclude_self=True):
    """
    Each edge present with prob p (independent).
    """
    A = (np.random.rand(n_pre, n_post) < p).astype(np.float32)
    if exclude_self and n_pre == n_post:
        np.fill_diagonal(A, 0)
    return A


def connect_distance(n_neurons, _, sigma=3.0, p_max=0.5, circular=False, exclude_self=True):
    """
    Distance-dependent binary connectivity (1D).
    - Neurons are placed on a 1D line with indices 0..n-1.
    - Connection prob decays with distance: p_ij = p_max * exp(-(d_ij^2)/(2*sigma^2))
    - If circular=True, distance wraps around (ring topology).

    Computes:
        A   : [n, n] binary adjacency (0/1)
        P   : [n, n] connection probabilities used
        D   : [n, n] pairwise distances
        
    Returns:
        A   : [n, n] binary adjacency (0/1)
    """
    idx = np.arange(n_neurons)
    D = np.abs(idx[:, None] - idx[None, :])  # |i-j|
    if circular:
        D = np.minimum(D, n_neurons - D)     # wrap-around distance on a ring

    # Pprobabilities decay exponentially with distance
    P = p_max * np.exp(-(D**2) / (10 * sigma**2))

    # Sample connections
    A = (np.random.rand(n_neurons, n_neurons) < P).astype(np.float32)

    if exclude_self:
        np.fill_diagonal(A, 0)
        A[np.diag_indices(n_neurons)] = 0.0

    return A