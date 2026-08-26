import numpy as np
import scipy as sp
from typing import Callable, List
from joblib import Parallel, delayed
import numba
import functools

def Effective_mapping(Hsys, S_list, omega, g, M, N):
    """
    Inputs:
        - System Hamiltonian Hsys (numpy array)
        - Coupling operators S (list of numpy arrays)
        - Frequencies of bosonic modes omega (list of floats)
        - Coupling strengths g (list of floats)
        - Number of bath bosonic modes M (int)
        - Output bosinic subspace dimension N (int)

    Outputs:
        - Effective Hamiltonian H_eff (numpy array)
        - Effective coupling operators S_eff (list of numpy arrays)

    Remarks:
        - Tensor product order is bath1 ⊗ bath2 ⊗ ... ⊗ bathN ⊗ system
        - Assumes all baths have the same number of modes M
        - M has to chosen to ensure convergence of the first N levels (might implement an adaptive scheme later)
        - I use the convetion where J_eff = J_RC and the extra epsilon factor is included in the coupling operators
    """
    
    if not isinstance(M, int) or M <= 0:
        raise ValueError("M must be a positive integer.")
    if not N < M:
        raise ValueError("N must be less than M.")
    
    # Number of baths
    N_bath = len(g)

    # Subspace dimensions
    dim_sys = Hsys.shape[0]
    dim_boson = M**N_bath

    # Construct bosonic operators in the total environment space
    def A(n):
        if n < 0 or n >= N_bath:
            raise ValueError("Bath index n is out of range.")
        a = np.zeros((M, M))
        for i in range(M - 1):
            a[i, i + 1] = np.sqrt(i + 1)
        # Construct the operator acting on the n-th bath
        ops = [np.eye(M)] * N_bath
        ops[n] = a
        return functools.reduce(np.kron, ops)
    def Adag(n):
        return A(n).conj().T

    # Construct the polaron unitary
    UP_terms = 0
    for m in range(N_bath):
        UP_terms += g[m] / omega[m] * np.kron( Adag(m) - A(m), S_list[m] )
    UP = sp.linalg.expm( UP_terms )

    # Construct the effective Hamiltonian and coupling operators
    H_eff = UP @ np.kron( np.eye(dim_boson), Hsys) @ UP.conj().T
    for m in range(N_bath):
        if not np.allclose(S_list[m] @ S_list[m], np.eye(dim_sys)):
            H_eff -= g[m]**2 / omega[m] * UP @ np.kron( np.eye(dim_boson), S_list[m] @ S_list[m] ) @ UP.conj().T
        if N > 1:
            H_eff += omega[m] * np.kron( Adag(m) @ A(m), np.eye(dim_sys))

    S_eff_list = [ -2*g[m]/omega[m] * UP @ np.kron( np.eye(dim_boson), S_list[m] ) @ UP.conj().T for m in range(N_bath) ]
    if N > 1:
        S_eff_list += [ np.kron( Adag(m) + A(m), np.eye(dim_sys) ) for m in range(N_bath) ]

    # Truncate to the first N env. levels
    dim_out = dim_sys*N**N_bath
    H_eff = H_eff[:dim_out, :dim_out]
    S_eff_list = [ S_eff_list[m][:dim_out, :dim_out] for m in range(N_bath) ]
    UP = UP[:dim_out, :dim_out]

    return H_eff, S_eff_list, UP

@numba.njit(parallel=True, fastmath=True, cache=True)
def _build_redfield_numba(Evals, S_eig, secular, tol_secular, gamma_values):
    N = Evals.shape[0]
    M = S_eig.shape[0]
    R = np.zeros((N * N, N * N, M + 1), dtype=np.complex128)

    def idx(m, n):
        return m * N + n
    
    deltaE = np.empty((N, N))
    for a in numba.prange(N):
        for b in range(N):
            deltaE[a, b] = Evals[a] - Evals[b]

    # main parallelized loops
    for ab in numba.prange(N * N):
        a = ab // N
        b = ab % N 

        idx_ab = ab
        delta_ab = deltaE[a, b]
        R[idx_ab, idx_ab, 0] += -1j * delta_ab

        for c in range(N):
            for d in range(N):
                for n in range(M):
                    # --- Term 1 ---
                    w_idx = d * N + c
                    G = gamma_values[w_idx]
                    tmp = 0.0 + 0.0j
                    tmp += S_eig[n, a, c] * S_eig[n, c, d] * G[n]
                    if (not secular) or (abs(delta_ab - deltaE[d, b]) <= tol_secular):
                        R[idx_ab, idx(d, b), n + 1] -= tmp

                    # --- Term 2 ---
                    w_idx = c * N + d
                    G = np.conj(gamma_values[w_idx])
                    tmp = 0.0 + 0.0j
                    tmp += S_eig[n, b, d] * S_eig[n, d, c] * G[n]
                    if (not secular) or (abs(delta_ab - deltaE[a, c]) <= tol_secular):
                        R[idx_ab, idx(a, c), n + 1] -= tmp

                    # --- Terms 3–4 (can reuse delta_ab again) ---
                    w_idx = c * N + a
                    G = gamma_values[w_idx]
                    tmp = 0.0 + 0.0j
                    tmp += S_eig[n, d, b] * S_eig[n, a, c] * G[n]
                    if (not secular) or (abs(delta_ab - deltaE[c, d]) <= tol_secular):
                        R[idx_ab, idx(c, d), n + 1] += tmp

                    w_idx = d * N + b
                    G = np.conj(gamma_values[w_idx])
                    tmp = 0.0 + 0.0j
                    tmp += S_eig[n, c, a] * S_eig[n, b, d] * G[n]
                    if (not secular) or (abs(delta_ab - deltaE[c, d]) <= tol_secular):
                        R[idx_ab, idx(c, d), n + 1] += tmp

    return R

def build_redfield_tensor(
    H: np.ndarray,
    S_list: List[np.ndarray],
    gamma_func: Callable[[float], np.ndarray],
    secular: bool = False,
    tol_secular: float = 1e-8,
    sum_over_baths: bool = True
) -> np.ndarray:

    # --- Setup ---
    H = np.asarray(H, dtype=np.complex128)
    N = H.shape[0]
    M = len(S_list)
    S_list = [np.asarray(S, dtype=np.complex128) for S in S_list]

    # --- Diagonalize Hamiltonian ---
    Evals, U = np.linalg.eigh(H)

    # Transform coupling operators
    S_eig = np.zeros((M, N, N), dtype=np.complex128)
    for a in range(M):
        S_eig[a] = U.conj().T @ S_list[a] @ U

    # Precompute transition frequencies and corresponding gamma values
    omega = Evals[:, None] - Evals[None, :]
    gamma_values = np.zeros((N * N, M), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            g = np.asarray(gamma_func(omega[i, j]), dtype=np.complex128)
            if g.ndim == 0:
                gamma_values[i * N + j, 0] = g
            else:
                for n in range(M):
                    gamma_values[i * N + j, n] = g[n % g.size]

    # --- Build Redfield tensor with Numba-accelerated core ---
    R = _build_redfield_numba(Evals, S_eig, secular, tol_secular, gamma_values)
    if sum_over_baths: R = np.sum(R, axis=2)

    return R,U

def evolve_density_matrix(R, time_steps, rho_init, U=None, n_jobs=-1, energy_basis=False):

    time_steps = np.array(time_steps)
    N = rho_init.shape[0]
    if U is None:
        U = np.eye(N)
    eigvals, eigvecs = np.linalg.eig(R)

    rho_energy = U.conj().T @ rho_init @ U
    rho_vec = rho_energy.reshape(-1)

    def evolve_single(t):
        U_t = eigvecs @ np.diag(np.exp(eigvals * t)) @ np.linalg.inv(eigvecs)
        rho_vec_t = U_t @ rho_vec
        rho_matrix_t = rho_vec_t.reshape(N, N)

        # ensure unital trace and hermiticity
        rho_matrix_t = 0.5 * (rho_matrix_t + rho_matrix_t.conj().T)
        rho_matrix_t /= np.trace(rho_matrix_t)

        # return to original basis
        if energy_basis:
            return rho_matrix_t
        else:
            return U @ rho_matrix_t @ U.conj().T

    rho_t = Parallel(n_jobs=n_jobs)(delayed(evolve_single)(t) for t in time_steps)
    return rho_t

def evolve_observables(R, time_steps, rho_init, observables, U=None, n_jobs=-1, energy_basis=False):

    # Parallelized density matrix evolution in site basis
    rho_t_list = evolve_density_matrix(R, time_steps, rho_init, U=U, n_jobs=n_jobs, energy_basis=energy_basis)

    # Pre-allocate results
    obs_names = list(observables.keys())
    results = {name: np.empty(len(time_steps), dtype=complex) for name in obs_names}

    # Parallelize over time steps
    def compute_expectations(rho_t):
        return [np.trace(rho_t @ observables[name]) for name in obs_names]

    expectations = Parallel(n_jobs=n_jobs)(
        delayed(compute_expectations)(rho_t)
        for rho_t in rho_t_list
    )

    # Convert to arrays
    for j, name in enumerate(obs_names):
        results[name] = np.array([vals[j] for vals in expectations])

    return results

def redfield_blocks(R, N, order='F'):
    """
    Split a Redfield tensor into population/coherence blocks.

    Parameters
    ----------
    R : (N^2, N^2) ndarray
        Redfield superoperator in Liouville space.
    N : int
        Hilbert space dimension.
    order : {'F','C'}, optional
        Vectorization convention:
        - 'F' = Fortran/column-major (used by QuTiP: index = i + j*N).
        - 'C' = C/row-major (index = j + i*N).

    Returns
    -------
    blocks : dict
        {
          'R_pp': pop<-pop      (N x N),
          'R_pc': pop<-coh      (N x (N^2-N)),
          'R_cp': coh<-pop      ((N^2-N) x N),
          'R_cc': coh<-coh      ((N^2-N) x (N^2-N)),
          'idx_pop': indices of populations in Liouville basis,
          'idx_coh': indices of coherences in Liouville basis
        }
    """
    idx_pop = []
    idx_coh = []
    for i in range(N):
        for j in range(N):
            if order == 'F':
                k = i + j*N   # column-major vec (QuTiP)
            else:
                k = j + i*N   # row-major
            if i == j:
                idx_pop.append(k)
            else:
                idx_coh.append(k)

    idx_pop = np.array(idx_pop, dtype=int)
    idx_coh = np.array(idx_coh, dtype=int)

    # Now all blocks have correct shapes:
    R_pp = R[np.ix_(idx_pop, idx_pop)]  # (N x N)
    R_pc = R[np.ix_(idx_pop, idx_coh)]  # (N x (N^2-N))
    R_cp = R[np.ix_(idx_coh, idx_pop)]  # ((N^2-N) x N)
    R_cc = R[np.ix_(idx_coh, idx_coh)]  # ((N^2-N) x (N^2-N))

    return {
        'R_pp': R_pp,
        'R_pc': R_pc,
        'R_cp': R_cp,
        'R_cc': R_cc,
        'idx_pop': idx_pop,
        'idx_coh': idx_coh
    }

def reduce_matrix_coordinate(M, k):
    """
    Reduce the linear system dx/dt = M x by eliminating coordinate x[k]
    using the constraint sum(x) = 1.

    Args:
        M : n×n numpy array
        k : index (0-based) of the coordinate to eliminate

    Returns:
        A_red : (n-1)×(n-1) reduced matrix
        b_red : (n-1) vector
        perm  : the permutation used so coordinates correspond as:
                y corresponds to all coordinates except k, in this order.
    """
    M = np.asarray(M)
    n = M.shape[0]

    # Build a permutation that moves index k to the last position
    perm = [i for i in range(n) if i != k] + [k]

    # Apply permutation: M' = P M P^{-1}
    Mp = M[np.ix_(perm, perm)]

    # Partition Mp as usual
    M11 = Mp[:n-1, :n-1]
    M12 = Mp[:n-1, n-1:n]      # (n-1)x1 column
    ones = np.ones((1, n-1))   # row vector

    # Reduced system
    A_red = M11 - M12 @ ones
    b_red = M12[:, 0]

    return A_red, b_red, perm

def steady_state_vec(R, H, tol=1e-12, reg=1e-14):
    """
    Solve for steady-state vector rho_ss_vec for d/dt rho = R rho.
    Uses the constraint Tr(rho) = 1 and solves the augmented system:
    [R ; trace_constraint] @ rho_vec = [0 ; 1] via least-squares.

    Args:
        R : (n^2, n^2) or (n^2, n^2, M) Redfield tensor
        tol : threshold for considering small singular values
        reg : small regularization added to diagonal if needed

    Returns:
        rho_ss_vec : length n^2 complex vector
    """

    R = np.sum(R, axis=2) if R.ndim == 3 else R
    dim = int(np.sqrt(R.shape[0]))

    # vectorized identity for trace constraint
    one_vec = np.eye(dim).reshape(-1)

    # zero vector for RHS in energy basis
    zero_state = np.zeros((dim, dim), dtype=complex)
    zero_state[0,0] = 1.0
    eigvals, U = np.linalg.eig(H)
    #zero_state = U.conj().T @ zero_state @ U
    zero_vec = zero_state.reshape(-1)

    L = R + np.outer(zero_vec, one_vec)
    rho_ss_vec, *_ = np.linalg.lstsq(L, zero_vec, rcond=None)
    
    return rho_ss_vec

def heat_current(R, H, U=None, bath_index=None, tol=1e-12, reg=1e-14):
    """
    Compute the heat current for a given Redfield tensor.
    
    The heat current is defined as J = Tr(L_alpha(rho_ss) H), where:
    - L_alpha is the Redfield superoperator contribution from bath alpha
    - rho_ss is the steady-state density matrix
    - H is the system Hamiltonian
    
    Parameters
    ----------
    R : (n^2, n^2) or (n^2, n^2, M) ndarray
        Redfield tensor. If 3D, R[:,:,bath_index] is extracted.
        If bath_index is None and R is 3D, sums over all baths.
    H : (n, n) ndarray
        System Hamiltonian (can be in any basis, will be diagonalized).
    bath_index : int, optional
        Index of the bath for which to compute heat current.
        Only used if R is 3D. If None and R is 3D, sums over all baths.
    tol : float, optional
        Tolerance for singular values in steady-state calculation.
    reg : float, optional
        Tikhonov regularization parameter for ill-conditioned systems.
    
    Returns
    -------
    J : float
        Heat current (real number).
    """

    if U is None:
        U = np.eye(H.shape[0])
    
    # Handle 3D tensor (per-bath decomposition)
    if R.ndim == 3:
        if bath_index is None:
            raise ValueError("specify bath index")
        else:
            if bath_index < 1 or bath_index >= R.shape[2]:
                raise ValueError(f"bath_index {bath_index} out of range [1, {R.shape[2]-1}]")
            R_alpha = R[:, :, bath_index]
    elif R.ndim == 2:
        raise ValueError("R must be a 3D array")
    
    dim = int(np.sqrt(R.shape[0]))
    
    # Compute steady state density matrix
    rho_ss_vec = steady_state_vec(R, H, tol=tol, reg=reg)
    
    # Apply Redfield superoperator to steady state vector
    L_rho_ss_vec = R_alpha @ rho_ss_vec
    L_rho_ss = L_rho_ss_vec.reshape((dim, dim))
    
    # unitary and Hs in energy eigenbasis
    eigvals, U_H = np.linalg.eigh(H)
    U = U_H @ U @ U_H.conj().T
    H = np.diag(eigvals)

    # Compute heat current: J = Tr(L(rho_ss) H)
    J = np.real(np.trace(U.conj().T @ L_rho_ss @ H @ U))
    
    return J
