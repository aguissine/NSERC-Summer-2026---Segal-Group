# =====================================
# Flux computation for configuration B
# =====================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ===========
# Parameters
# ===========

J_max = 1
alpha = 3

# ================
# Configuration B
# ================

R = np.array([[ 26.42338043,  -7.94752626],
              [ 23.5676572 , -10.82584156],
              [ 12.9945502 , -15.04985626],
              [ 10.04355896, -17.75500788],
              [ 22.09827259,  11.50615309],
              [  4.77769072, -29.50708222],
              [  3.61769485,  13.64602815],
              [-13.15603736,  -3.06908283],
              [ -7.72037927,  25.74900317],
              [-22.17787943,  18.52470886]])

# --------------------------
# Plot of the configuration 
# --------------------------

plt.figure()
plt.scatter(R[:, 0], R[:, 1])

for i, (x, y) in enumerate(R):
    plt.text(x, y, str(i), ha='right', va='bottom')

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# ============
# Hamiltonian
# ============

N = R.shape[0]
H = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i != j:
            dist = np.linalg.norm(R[i] - R[j])
            H[i, j] = J_max / dist**alpha

# ==============
# Flux function
# ==============

def J_mn(n, m, Gamma, gamma_l):

    #--------------------
    # Unitary Liouvillian
    #--------------------

    I = np.eye(N)
    L_unitary = -1j*(np.kron(I,H)-np.kron(H.T,I))

    #------------------
    # On-site dephasing
    #------------------

    L_diss = np.zeros((N*N, N*N), dtype=complex)

    for j in range(N):
        Lj = np.zeros((N, N))
        Lj[j, j] = 1.0
        Lj_dagger = np.transpose(np.conjugate(Lj))
        
        Lj2 = Lj_dagger @ Lj

        L_diss += (
            np.kron(np.conjugate(Lj), Lj)
            - 0.5 * np.kron(I, Lj2)
            - 0.5 * np.kron(Lj2.T, I)
        )

    L_diss *= Gamma

    #--------------------------------
    # Phenomenological loss at site m
    #--------------------------------

    def idx(i, j):
        return i + j * N   

    L_loss = np.zeros((N*N, N*N), dtype=complex)
    L_loss[idx(m, m), idx(m, m)] -= gamma_l

    for i in range(N):
        if i != m:
            L_loss[idx(m, i), idx(m, i)] -= gamma_l / 2
            L_loss[idx(i, m), idx(i, m)] -= gamma_l / 2

    #------------------
    # Total Liouvillian
    #------------------

    L = L_unitary + L_diss + L_loss

    #-------------------------------------------------------------
    # Enforce trace = 1 (NESS condition) at the injection site eqn
    #-------------------------------------------------------------

    vec_I = np.eye(N).reshape(N*N, order="F")

    L[int(11*n), :] = vec_I #---------> Replacing injection row with identity
    #L[0,:] = vec_I #---------> Replacing first row with identity
    
    b = np.zeros(N*N)
    b[int(11*n)] = 1.0
    #b[0] = 1.0


    rho_vec = np.linalg.solve(L, b)
    rho_NESS = rho_vec.reshape((N, N), order="F")

    # Optional sanity checks
    #print("Trace:", np.trace(rho_NESS))
    #print("Hermiticity error:", np.linalg.norm(rho_NESS - rho_NESS.conj().T))

    return 2*H*np.imag(rho_NESS)

# =================
# Plotting results
# =================

matrixA = np.abs(J_mn(5, 8, 1e-2, 1e-5))

matrixB = np.abs(J_mn(5, 8, 1e-4, 1e-5))

fig, ax = plt.subplots()

# Display the matrix
im = ax.imshow(matrixB, origin='upper', cmap='viridis')

# Set ticks at the center of each pixel
ax.set_xticks(np.arange(matrixB.shape[1]))
ax.set_yticks(np.arange(matrixB.shape[0]))

# Label ticks from 1 to 10
ax.set_xticklabels(np.arange(0, 10))
ax.set_yticklabels(np.arange(0, 10))

# Move x-axis ticks to the top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Optional: grid lines to see the blocks clearly
ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1)
ax.tick_params(which='minor', bottom=False, left=False)

plt.colorbar(im)
plt.show()

np.save('matrixA.npy',matrixA)
np.save('matrixB.npy',matrixB)
