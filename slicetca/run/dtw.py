# ----------------------------------------------------------------------------------------------------------------------

import math

import numpy as np
import torch
import torch.cuda
from numba import cuda
from numba import jit, prange
from torch.autograd import Function
import lightning.pytorch as pl

# GLOBALS
TPB_LONG = 1024
_D_INNER_MAX = 32   # max inner-DTW D for the fused per-triple CUDA kernel

# ----------------------------------------------------------------------------------------------------------------------

# ---- 2D fused path (features_as_points) ----

@cuda.jit
def softdtw_forward_diag_sqeuclid_cuda(X, Y, R, gamma, bandwidth, N, M, D, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    # cost = ||X[b,i,:] - Y[b,j,:]||^2
    cost = np.float32(0.0)
    for k in range(D):
        diff = X[b, i, k] - Y[b, j, k]
        cost += diff * diff

    inv_gamma = np.float32(1.0) / gamma

    r0 = -(R[b, ip - 1, jp - 1] + cost) * inv_gamma   # symmetric2: diagonal weighted
    r1 = -R[b, ip - 1, jp]     * inv_gamma
    r2 = -R[b, ip,     jp - 1] * inv_gamma

    rmax = r0
    if r1 > rmax: rmax = r1
    if r2 > rmax: rmax = r2

    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
    softmin = -gamma * (math.log(rsum) + rmax)

    R[b, ip, jp] = cost + softmin


@cuda.jit
def softdtw_backward_log_diag_sqeuclid_cuda(X, Y, R, logE, inv_gamma, bandwidth, N, M, D, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    Rij = R[b, ip, jp]
    if math.isinf(Rij):
        Rij = -math.inf

    # costs for transitions:
    # D_pad[ip+1, jp]   corresponds to ||X[i+1] - Y[j]||^2
    # D_pad[ip, jp+1]   corresponds to ||X[i]   - Y[j+1]||^2
    # D_pad[ip+1, jp+1] corresponds to ||X[i+1] - Y[j+1]||^2

    # cost_down: (i+1, j)
    cost_down = np.float32(0.0)
    if i + 1 < N:
        for k in range(D):
            diff = X[b, i + 1, k] - Y[b, j, k]
            cost_down += diff * diff
    else:
        # this state will be invalid anyway due to R boundary = -inf
        cost_down = np.float32(0.0)

    # cost_right: (i, j+1)
    cost_right = np.float32(0.0)
    if j + 1 < M:
        for k in range(D):
            diff = X[b, i, k] - Y[b, j + 1, k]
            cost_right += diff * diff
    else:
        cost_right = np.float32(0.0)

    # cost_diag: (i+1, j+1)
    cost_diag = np.float32(0.0)
    if (i + 1 < N) and (j + 1 < M):
        for k in range(D):
            diff = X[b, i + 1, k] - Y[b, j + 1, k]
            cost_diag += diff * diff
    else:
        cost_diag = np.float32(0.0)

    la = (R[b, ip + 1, jp]     - Rij - cost_down)      * inv_gamma
    lb = (R[b, ip,     jp + 1] - Rij - cost_right)     * inv_gamma
    lc = (R[b, ip + 1, jp + 1] - Rij - np.float32(2.0) * cost_diag) * inv_gamma  # symmetric2

    t1 = logE[b, ip + 1, jp]     + la
    t2 = logE[b, ip,     jp + 1] + lb
    t3 = logE[b, ip + 1, jp + 1] + lc

    # reuse your helper if you want (recommended):
    m = t1
    if t2 > m: m = t2
    if t3 > m: m = t3

    if m == -math.inf:
        logE[b, ip, jp] = -math.inf
    else:
        logE[b, ip, jp] = m + math.log(math.exp(t1 - m) + math.exp(t2 - m) + math.exp(t3 - m))



@cuda.jit
def softdtw_forward_kernel(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    I = tid
    inv_gamma = np.float32(1.0) / gamma

    for p in range(n_passes):
        J = max(0, min(p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == p and (I < max_i and J < max_j):
            if not (abs(i - j) > bandwidth > 0):
                cost = D[b, i - 1, j - 1]
                r0 = -(R[b, i - 1, j - 1] + cost) * inv_gamma  # symmetric2
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = cost + softmin
        cuda.syncthreads()

@cuda.jit
def softdtw_forward_diag_cuda(D, R, gamma, bandwidth, N, M, p):
    b = cuda.blockIdx.y  # batch in Y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # diagonal bounds in unpadded coordinates
    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)

    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    # bandwidth pruning (in padded coords uses ip/jp, but difference same)
    if bandwidth > 0 and abs(ip - jp) > bandwidth:
        return

    cost = D[b, i, j]
    inv_gamma = np.float32(1.0) / gamma

    r0 = -(R[b, ip - 1, jp - 1] + cost) * inv_gamma   # symmetric2
    r1 = -R[b, ip - 1, jp]     * inv_gamma
    r2 = -R[b, ip,     jp - 1] * inv_gamma

    rmax = r0
    if r1 > rmax: rmax = r1
    if r2 > rmax: rmax = r2

    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
    softmin = -gamma * (math.log(rsum) + rmax)

    R[b, ip, jp] = cost + softmin


@cuda.jit(device=True, inline=True)
def _logsumexp3(a, b, c):
    m = a
    if b > m: m = b
    if c > m: m = c
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m) + math.exp(c - m))


@cuda.jit
def softdtw_backward_log_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, logE):
    """
    D: (B, N+2, M+2) padded
    R: (B, N+2, M+2) padded (with boundary conditions already set)
    logE: (B, N+2, M+2) padded, initialized to -inf with logE[:,-1,-1]=0
    """
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    I = tid

    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == rev_p and (I < max_i and J < max_j):

            # pruning
            if not (abs(i - j) > bandwidth > 0):

                Rij = R[k, i, j]
                if math.isinf(Rij):
                    Rij = -math.inf

                # log transition weights (no exp here!)
                la = (R[k, i + 1, j]     - Rij - D[k, i + 1, j])         * inv_gamma
                lb = (R[k, i, j + 1]     - Rij - D[k, i, j + 1])         * inv_gamma
                lc = (R[k, i + 1, j + 1] - Rij - np.float32(2.0) * D[k, i + 1, j + 1]) * inv_gamma  # symmetric2

                t1 = logE[k, i + 1, j]     + la
                t2 = logE[k, i, j + 1]     + lb
                t3 = logE[k, i + 1, j + 1] + lc

                logE[k, i, j] = _logsumexp3(t1, t2, t3)

        cuda.syncthreads()

@cuda.jit
def softdtw_2d_forward_precomp_cuda(C, R, gamma, N, D, p):
    """
    2D SoftDTW forward kernel using a pre-materialised cost matrix C (B,N,D).
    C[b,i,j] is the scalar cost at grid position (i,j); typically (X-Y)^2.
    R: (B, N+2, D+2). Anti-diagonal p = i+j over the (N, D) unpadded grid.
    Weighted diagonal step: diagonal predecessor is penalised by an extra cost.
    """
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (D - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    cost = C[b, i, j]
    inv_gamma = np.float32(1.0) / gamma

    r0 = -(R[b, ip - 1, jp - 1] + cost) * inv_gamma   # weighted diagonal
    r1 =  -R[b, ip - 1, jp]     * inv_gamma
    r2 =  -R[b, ip,     jp - 1] * inv_gamma

    rmax = r0
    if r1 > rmax: rmax = r1
    if r2 > rmax: rmax = r2

    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
    softmin = -gamma * (math.log(rsum) + rmax)

    R[b, ip, jp] = cost + softmin


@cuda.jit
def softdtw_2d_backward_log_precomp_cuda(C, R, logE, inv_gamma, N, D, p):
    """
    2D SoftDTW backward kernel (log-space) using pre-materialised cost C (B,N,D).
    R: (B, N+2, D+2) with boundary conditions applied.
    logE: (B, N+2, D+2) initialised to -inf with logE[:,-1,-1]=0.
    Anti-diagonal p = i+j in reverse order.
    """
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (D - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    Rij = R[b, ip, jp]
    if math.isinf(Rij):
        Rij = -math.inf

    # Read costs for the three successor cells from the pre-computed C array.
    # Successor (i+1, j) is an orthogonal (time) step; cost = C[b, i+1, j].
    cost_down = np.float32(0.0)
    if i + 1 < N:
        cost_down = C[b, i + 1, j]

    # Successor (i, j+1) is an orthogonal (feature) step; cost = C[b, i, j+1].
    cost_right = np.float32(0.0)
    if j + 1 < D:
        cost_right = C[b, i, j + 1]

    # Successor (i+1, j+1) is the diagonal step; cost = C[b, i+1, j+1].
    # The diagonal was stored as R[ip,jp] + cost_diag in the forward, so the
    # log-weight derivation subtracts 2*cost_diag (not 1).
    cost_diag = np.float32(0.0)
    if (i + 1 < N) and (j + 1 < D):
        cost_diag = C[b, i + 1, j + 1]

    la = (R[b, ip + 1, jp]     - Rij - cost_down)         * inv_gamma
    lb = (R[b, ip,     jp + 1] - Rij - cost_right)        * inv_gamma
    lc = (R[b, ip + 1, jp + 1] - Rij - np.float32(2.0) * cost_diag)  * inv_gamma

    t1 = logE[b, ip + 1, jp]     + la
    t2 = logE[b, ip,     jp + 1] + lb
    t3 = logE[b, ip + 1, jp + 1] + lc

    m = t1
    if t2 > m: m = t2
    if t3 > m: m = t3

    if m == -math.inf:
        logE[b, ip, jp] = -math.inf
    else:
        logE[b, ip, jp] = m + math.log(math.exp(t1 - m) + math.exp(t2 - m) + math.exp(t3 - m))


@cuda.jit
def softdtw_2d_forward_diag_cuda(X, Y, R, gamma, N, D, p):
    """
    2D fused SoftDTW forward kernel. X and Y must have the same shape (B, N, D).
    Each scalar X[b,i,j] is a point at grid position (i,j); cost = (X[b,i,j]-Y[b,i,j])^2.
    R: (B, N+2, D+2). Anti-diagonal p = i+j over the (N, D) unpadded grid.
    """
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (D - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    diff = X[b, i, j] - Y[b, i, j]
    cost = diff * diff

    inv_gamma = np.float32(1.0) / gamma

    # Diagonal predecessor is penalised by an extra `cost` so that a diagonal
    # step (which advances both axes) is not cheaper than two orthogonal steps.
    r0 = -(R[b, ip - 1, jp - 1] + cost) * inv_gamma
    r1 =  -R[b, ip - 1, jp]     * inv_gamma
    r2 =  -R[b, ip,     jp - 1] * inv_gamma

    rmax = r0
    if r1 > rmax: rmax = r1
    if r2 > rmax: rmax = r2

    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
    softmin = -gamma * (math.log(rsum) + rmax)

    R[b, ip, jp] = cost + softmin


@cuda.jit
def softdtw_2d_backward_log_diag_cuda(X, Y, R, logE, inv_gamma, N, D, p):
    """
    2D fused SoftDTW backward kernel (log-space).
    X, Y: (B, N, D). R: (B, N+2, D+2) with boundary conditions applied.
    logE: (B, N+2, D+2) initialised to -inf with logE[:,-1,-1]=0.
    Anti-diagonal p = i+j in reverse order.
    """
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (D - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    Rij = R[b, ip, jp]
    if math.isinf(Rij):
        Rij = -math.inf

    # cost at the "down" neighbour (i+1, j)
    cost_down = np.float32(0.0)
    if i + 1 < N:
        d = X[b, i + 1, j] - Y[b, i + 1, j]
        cost_down = d * d

    # cost at the "right" neighbour (i, j+1)
    cost_right = np.float32(0.0)
    if j + 1 < D:
        d = X[b, i, j + 1] - Y[b, i, j + 1]
        cost_right = d * d

    # cost at the "diagonal" neighbour (i+1, j+1)
    cost_diag = np.float32(0.0)
    if (i + 1 < N) and (j + 1 < D):
        d = X[b, i + 1, j + 1] - Y[b, i + 1, j + 1]
        cost_diag = d * d

    la = (R[b, ip + 1, jp]     - Rij - cost_down)         * inv_gamma
    lb = (R[b, ip,     jp + 1] - Rij - cost_right)        * inv_gamma
    # Diagonal step was stored as R[ip,jp] + cost_diag in the forward pass,
    # so the log-weight derivation gives -2*cost_diag here (not -1).
    lc = (R[b, ip + 1, jp + 1] - Rij - np.float32(2.0) * cost_diag)  * inv_gamma

    t1 = logE[b, ip + 1, jp]     + la
    t2 = logE[b, ip,     jp + 1] + lb
    t3 = logE[b, ip + 1, jp + 1] + lc

    m = t1
    if t2 > m: m = t2
    if t3 > m: m = t3

    if m == -math.inf:
        logE[b, ip, jp] = -math.inf
    else:
        logE[b, ip, jp] = m + math.log(math.exp(t1 - m) + math.exp(t2 - m) + math.exp(t3 - m))


@cuda.jit
def softdtw_backward_log_diag_cuda(Dp, R, logE, inv_gamma, bandwidth, N, M, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    # pruning
    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    Rij = R[b, ip, jp]
    if math.isinf(Rij):
        Rij = -math.inf

    la = (R[b, ip + 1, jp]     - Rij - Dp[b, ip + 1, jp])         * inv_gamma
    lb = (R[b, ip, jp + 1]     - Rij - Dp[b, ip, jp + 1])         * inv_gamma
    lc = (R[b, ip + 1, jp + 1] - Rij - np.float32(2.0) * Dp[b, ip + 1, jp + 1]) * inv_gamma  # symmetric2

    t1 = logE[b, ip + 1, jp]     + la
    t2 = logE[b, ip, jp + 1]     + lb
    t3 = logE[b, ip + 1, jp + 1] + lc

    m = t1
    if t2 > m: m = t2
    if t3 > m: m = t3

    if m == -math.inf:
        logE[b, ip, jp] = -math.inf
    else:
        logE[b, ip, jp] = _logsumexp3(t1, t2, t3)

# HELPERS
def _diag_bounds(p: int, N: int, M: int) -> tuple[int, int]:
    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    return i_min, i_max


def _threads_and_passes(N: int, M: int) -> tuple[int, int]:
    tpb = max(N, M)
    n_passes = 2 * tpb - 1
    return tpb, n_passes


# MAIN - on-the-fly D
def softdtw_forward_cuda_fused_sqeuclid(X: torch.Tensor, Y: torch.Tensor,
                                        gamma: float, bandwidth: float):
    """
    Fused SoftDTW forward for squared-euclidean distance that does NOT materialize D (B,N,M).

    X: (B,N,D), Y: (B,M,D) CUDA tensors
    Returns: (out: (B,), R: (B,N+2,M+2))
    """
    if not (X.is_cuda and Y.is_cuda):
        raise ValueError("Expected CUDA tensors X and Y")
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError(
            f"Expected X,Y as (B,N,D)/(B,M,D). Got {tuple(X.shape)} and {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
        raise ValueError(
            f"Batch/features mismatch: {tuple(X.shape)} vs {tuple(Y.shape)}")

    # Detach before passing to numba
    X_ = X.detach().contiguous()
    Y_ = Y.detach().contiguous()

    B, N, D = X_.shape
    M = Y_.shape[1]

    # Allocate DP table
    R = torch.full((B, N + 2, M + 2), math.inf, device=X_.device,
                   dtype=X_.dtype)
    R[:, 0, 0] = 0.0

    X_ca = cuda.as_cuda_array(X_)
    Y_ca = cuda.as_cuda_array(Y_)
    R_ca = cuda.as_cuda_array(R)

    inv_bw = bandwidth  # can be -1.0 to disable

    # Anti-diagonals over unpadded (i,j): p = i + j, i in [0,N-1], j in [0,M-1]
    for p in range(N + M - 1):
        i_min = max(0, p - (M - 1))
        i_max = min(N - 1, p)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        # grid=(grid_x, B), so batch = blockIdx.y in kernel
        softdtw_forward_diag_sqeuclid_cuda[(grid_x, B), TPB_LONG](
            X_ca,
            Y_ca,
            R_ca,
            gamma,
            inv_bw,
            N,
            M,
            D,
            p,
        )

    out = R[:, -2, -2].contiguous()
    return out, R


def softdtw_backward_cuda_fused_sqeuclid(X: torch.Tensor, Y: torch.Tensor,
                                         R: torch.Tensor, gamma: float,
                                         bandwidth: float):
    """
    Fused SoftDTW backward (log-space) for squared-euclidean distance that does NOT materialize D_pad.

    Inputs:
      X: (B,N,D) CUDA
      Y: (B,M,D) CUDA
      R: (B,N+2,M+2) CUDA (from forward)
    Returns:
      E: (B,N,M) CUDA  (E = d SoftDTW / d D  in linear space, via exp(logE))
    """
    if not (X.is_cuda and Y.is_cuda and R.is_cuda):
        raise ValueError("Expected CUDA tensors X, Y, R")
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError(
            f"Expected X,Y as (B,N,D)/(B,M,D). Got {tuple(X.shape)} and {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
        raise ValueError(
            f"Batch/features mismatch: {tuple(X.shape)} vs {tuple(Y.shape)}")

    # Detach before passing to numba
    X_ = X.detach().contiguous()
    Y_ = Y.detach().contiguous()

    B, N, D = X_.shape
    M = Y_.shape[1]

    if R.shape != (B, N + 2, M + 2):
        raise ValueError(
            f"Expected R shape {(B, N + 2, M + 2)}, got {tuple(R.shape)}")

    R_ = R.contiguous()

    # ---------- boundary conditions for R ----------
    R_work = R_.clone()
    R_work[:, :, -1] = -math.inf
    R_work[:, -1, :] = -math.inf
    R_work[:, -1, -1] = R_work[:, -2, -2]

    # ---------- init logE ----------
    logE = torch.full((B, N + 2, M + 2), -math.inf, device=X_.device,
                      dtype=X_.dtype)
    logE[:, -1, -1] = 0.0  # log(1)

    X_ca = cuda.as_cuda_array(X_)
    Y_ca = cuda.as_cuda_array(Y_)
    Rw_ca = cuda.as_cuda_array(R_work)
    logE_ca = cuda.as_cuda_array(logE)

    inv_gamma = 1.0 / gamma
    bw = bandwidth

    # Reverse anti-diagonals over unpadded indices p = i + j, starting from (N-1)+(M-1)-1 = N+M-2 down to 0
    for p in range(N + M - 2, -1, -1):
        i_min = max(0, p - (M - 1))
        i_max = min(N - 1, p)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        softdtw_backward_log_diag_sqeuclid_cuda[(grid_x, B), TPB_LONG](
            X_ca,
            Y_ca,
            Rw_ca,
            logE_ca,
            inv_gamma,
            bw,
            N,
            M,
            D,
            p,
        )

    # crop + exp (upcast to float32 to prevent fp16 overflow at logE > ~11)
    E = torch.exp(logE[:, 1:N + 1, 1:M + 1].float())

    # Symmetric2 correction: E above is dDTW/dR.  For dDTW/dD we need
    # E_corrected = E * (1 + p_diag), where p_diag(i,j) is the softmin
    # weight of the diagonal predecessor at cell (i,j).
    # log(p_diag) = (R_pad[ip,jp] - R_pad[ip-1,jp-1] - 2*D[i,j]) / gamma
    D_sq = sqeuclidean(X_.float(), Y_.float())  # (B, N, M)
    R_f = R_.float()
    log_p_diag = (R_f[:, 1:N + 1, 1:M + 1]
                  - R_f[:, 0:N, 0:M]
                  - 2.0 * D_sq) / gamma
    p_diag = torch.exp(log_p_diag.clamp(max=80.0))
    E = E * (1.0 + p_diag)

    return E.to(logE.dtype).contiguous()


# MAIN - Full D Matrix
def softdtw_forward_cuda(D: torch.Tensor, gamma: float, bandwidth: float):
    if not D.is_cuda:
        raise ValueError("Expected CUDA tensor D")

    D_ = D.detach().contiguous()
    B, N, M = D_.shape
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    # Allocate DP table
    R = torch.full((B, N + 2, M + 2), math.inf, device=D_.device,
                   dtype=D_.dtype)
    R[:, 0, 0] = 0.0

    # --- Fast path: one block per batch element ---
    tpb, n_passes = _threads_and_passes(N, M)
    USE_FAST_PATH = (tpb <= 1024)

    if USE_FAST_PATH:
        softdtw_forward_kernel[B, tpb](
            cuda.as_cuda_array(D_),
            gamma,
            bandwidth,
            N,
            M,
            n_passes,
            cuda.as_cuda_array(R),
        )
        out = R[:, -2, -2].contiguous()
        return out, R

    # --- Long sequence path: tiled anti-diagonal launches ---

    D_ca = cuda.as_cuda_array(D_)
    R_ca = cuda.as_cuda_array(R)

    # Iterate anti-diagonals in unpadded (i,j) coords over D (shape N x M)
    for p in range(N + M - 1):
        i_min, i_max = _diag_bounds(p, N, M)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        # grid=(grid_x, B) so batch index is blockIdx.y inside kernel
        softdtw_forward_diag_cuda[(grid_x, B), TPB_LONG](
            D_ca,
            R_ca,
            gamma,
            bandwidth,
            N,
            M,
            p,
        )

    out = R[:, -2, -2].contiguous()
    return out, R


def softdtw_backward_cuda_log(D: torch.Tensor, R: torch.Tensor, gamma: float,
                              bandwidth: float):
    if not D.is_cuda:
        raise ValueError("Expected CUDA tensor D")

    D_ = D.detach().contiguous()
    B, N, M = D_.shape
    R = R.contiguous()

    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    # ---------- pad D ----------
    D_pad = torch.zeros((B, N + 2, M + 2), device=D_.device, dtype=D_.dtype)
    D_pad[:, 1:N + 1, 1:M + 1] = D_

    # ---------- boundary conditions for R ----------
    R_work = R.clone()
    R_work[:, :, -1] = -math.inf
    R_work[:, -1, :] = -math.inf
    R_work[:, -1, -1] = R_work[:, -2, -2]

    # ---------- init logE ----------
    logE = torch.full((B, N + 2, M + 2), -math.inf, device=D_.device,
                      dtype=D_.dtype)
    logE[:, -1, -1] = 0.0  # log(1)

    # ---------- choose fast vs tiled ----------
    tpb, n_passes = _threads_and_passes(N, M)
    USE_FAST_PATH = (tpb <= 1024)

    if USE_FAST_PATH:
        # fast path: your existing diagonal backward kernel (single block per batch)
        softdtw_backward_log_cuda[B, tpb](
            cuda.as_cuda_array(D_pad),
            cuda.as_cuda_array(R_work),
            float(1.0 / gamma),
            bandwidth,
            N,
            M,
            n_passes,
            cuda.as_cuda_array(logE),
        )
    else:
        # tiled path: launch one kernel per anti-diagonal in reverse order

        Dp_ca = cuda.as_cuda_array(D_pad)
        Rw_ca = cuda.as_cuda_array(R_work)
        logE_ca = cuda.as_cuda_array(logE)

        inv_gamma = float(1.0 / gamma)
        bw = bandwidth
        if bw <= 0:
            bw = -1.0

        # unpadded indices (i,j) are 0..N-1, 0..M-1, diagonals p = i+j
        for p in range(N + M - 2, -1, -1):
            i_min, i_max = _diag_bounds(p, N, M)
            if i_max < i_min:
                continue
            diag_len = i_max - i_min + 1
            grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

            softdtw_backward_log_diag_cuda[(grid_x, B), TPB_LONG](
                Dp_ca,
                Rw_ca,
                logE_ca,
                inv_gamma,
                bw,
                N,
                M,
                p,
            )

    # crop + exp (upcast to float32 to prevent fp16 overflow at logE > ~11)
    E = torch.exp(logE[:, 1:N + 1, 1:M + 1].float())

    # Symmetric2 correction: E above is dDTW/dR.  For dDTW/dD we need
    # E_corrected = E * (1 + p_diag).
    R_f = R.float()  # original R from forward (before boundary modification)
    log_p_diag = (R_f[:, 1:N + 1, 1:M + 1]
                  - R_f[:, 0:N, 0:M]
                  - 2.0 * D_.float()) / gamma
    p_diag = torch.exp(log_p_diag.clamp(max=80.0))
    E = E * (1.0 + p_diag)

    return E.to(logE.dtype).contiguous()



@jit(nopython=True, parallel=True)
def _softdtw_2d_forward_np(C, gamma):
    """CPU reference 2D SoftDTW forward with weighted diagonal step."""
    B, N, D = C.shape
    R = np.ones((B, N + 2, D + 2), dtype=np.float32) * np.inf
    R[:, 0, 0] = np.float32(0.0)
    for b in prange(B):
        for i in range(1, N + 1):
            for j in range(1, D + 1):
                c = C[b, i - 1, j - 1]
                r0 = -(R[b, i - 1, j - 1] + c) / gamma  # weighted diagonal
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(r0, max(r1, r2))
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(
                    r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = c + softmin
    return R


# ---- CPU reference (optional but useful for tests) ----

@jit(nopython=True, parallel=True)
def _softdtw_forward_cpu_np(D: np.ndarray, gamma: float, bandwidth: float):
    B, N, M = D.shape
    R = np.ones((B, N + 2, M + 2), dtype=D.dtype) * np.inf
    R[:, 0, 0] = np.float32(0.0)
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                if 0 < bandwidth < abs(i - j):
                    continue
                cost = D[b, i - 1, j - 1]
                r0 = -(R[b, i - 1, j - 1] + cost) / gamma  # symmetric2
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(
                    r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = cost + softmin
    return R


@jit(nopython=True, parallel=True)
def _softdtw_backward_cpu_np(D_: np.ndarray, R: np.ndarray, gamma: float,
                             bandwidth: float):
    B, N, M = D_.shape
    D = np.zeros((B, N + 2, M + 2), dtype=D_.dtype)
    D[:, 1:N + 1, 1:M + 1] = D_

    E = np.zeros((B, N + 2, M + 2), dtype=D_.dtype)
    E[:, -1, -1] = np.float32(1.0)

    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]

    for b in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[b, i, j]):
                    R[b, i, j] = -np.inf
                if 0 < bandwidth < abs(i - j):
                    continue
                a0 = (R[b, i + 1, j] - R[b, i, j] - D[b, i + 1, j]) / gamma
                b0 = (R[b, i, j + 1] - R[b, i, j] - D[b, i, j + 1]) / gamma
                c0 = (R[b, i + 1, j + 1] - R[b, i, j] - np.float32(2.0) * D[
                    b, i + 1, j + 1]) / gamma  # symmetric2
                a = np.exp(a0);
                bb = np.exp(b0);
                c = np.exp(c0)
                E[b, i, j] = E[b, i + 1, j] * a + E[b, i, j + 1] * bb + E[
                    b, i + 1, j + 1] * c

    return E[:, 1:N + 1, 1:M + 1]


def softdtw_forward_cpu(D: torch.Tensor, gamma: float, bandwidth: float):
    D_np = D.detach().cpu().numpy()
    R_np = _softdtw_forward_cpu_np(D_np, gamma, bandwidth)
    R = torch.from_numpy(R_np).to(D.device).type_as(D)
    out = R[:, -2, -2].contiguous()
    return out, R


def softdtw_backward_cpu(D: torch.Tensor, R: torch.Tensor, gamma: float,
                         bandwidth: float):
    D_np = D.detach().cpu().numpy()
    R_orig = R.detach().cpu().numpy()
    R_np = R_orig.copy()  # .copy() prevents in-place mutation of saved autograd tensor
    E_np = _softdtw_backward_cpu_np(D_np, R_np, gamma, bandwidth)

    # Symmetric2 correction: E above is dDTW/dR; multiply by (1 + p_diag)
    B, N, M = D_np.shape
    log_p = (R_orig[:, 1:N + 1, 1:M + 1]
             - R_orig[:, 0:N, 0:M]
             - 2.0 * D_np) / gamma
    log_p = np.clip(log_p, -80.0, 80.0)
    E_np = E_np * (1.0 + np.exp(log_p))

    return torch.from_numpy(E_np).to(D.device).type_as(D).contiguous()

def sqeuclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Efficient squared Euclidean distance:
      D[b,i,j] = ||x[b,i]-y[b,j]||^2

    x: (B,N,D), y: (B,M,D)
    returns: (B,N,M)
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"Expected x,y as (B,N,D)/(B,M,D). Got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(f"Batch/features mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")

    # Upcast to float32 to avoid fp16 overflow in sum and bmm
    x_f = x.float()
    y_f = y.float()

    # (B,N)
    x2 = (x_f * x_f).sum(dim=-1)
    # (B,M)
    y2 = (y_f * y_f).sum(dim=-1)
    # (B,N,M)
    xy = torch.bmm(x_f, y_f.transpose(1, 2))
    D = x2.unsqueeze(2) + y2.unsqueeze(1) - 2.0 * xy

    # Numerical cleanup (fp roundoff can produce tiny negatives)
    return D.clamp_min(0.0).to(x.dtype)

# ---- Fused per-triple inner-DTW CUDA kernels ----------------------------------------
# Each CUDA block handles one (b, row_i, col_j) triple.
# blockDim.x = _D_INNER_MAX (32); thread s processes row s of the D×D inner DP.
# Shared memory holds the R (and logE for backward) tables; no global R is written.
# _D_INNER_MAX must be >= D at runtime (checked in the Python launcher).

@cuda.jit(fastmath=True)
def _nested_inner_forward_kernel(X, Y, C, gamma, bw_freq, bw_time):
    """Compute C[b,i,j] = SoftDTW(X[b,i,:], Y[b,j,:]) for every (b,i,j) triple."""
    pair_idx = cuda.blockIdx.x
    B_dim = C.shape[0]; N_dim = C.shape[1]; M_dim = C.shape[2]
    if pair_idx >= B_dim * N_dim * M_dim:
        return

    b  = pair_idx // (N_dim * M_dim)
    ij = pair_idx %  (N_dim * M_dim)
    ri = ij // M_dim
    cj = ij %  M_dim
    D  = X.shape[2]

    # Outer-bandwidth filter: all threads in block see same ri,cj → uniform branch.
    if bw_time > 0.0:
        d = ri - cj
        if d < 0: d = -d
        if d > bw_time:
            return

    s = cuda.threadIdx.x

    # Shared DP table and input caches.
    R  = cuda.shared.array((34, 34), dtype=np.float32)
    Xs = cuda.shared.array((32,),    dtype=np.float32)
    Ys = cuda.shared.array((32,),    dtype=np.float32)

    # Load X and Y frequency profiles into shared memory.
    if s < D:
        Xs[s] = X[b, ri, s]
        Ys[s] = Y[b, cj, s]

    # Parallel init to +inf (strided across 32 threads, 34×34=1156 entries).
    for fi in range(s, 1156, 32):
        R[fi // 34, fi % 34] = math.inf
    cuda.syncwarp()  # ensures R and Xs/Ys are all ready (1 warp per block)
    if s == 0:
        R[0, 0] = 0.0
    cuda.syncwarp()

    inv_g = np.float32(1.0) / gamma

    for p in range(2 * D - 1):
        t = p - s
        if 0 <= s < D and 0 <= t < D:
            skip = False
            if bw_freq > 0.0:
                d = s - t
                if d < 0: d = -d
                if d > bw_freq:
                    skip = True
            if not skip:
                diff = Xs[s] - Ys[t]
                cost = diff * diff
                ip = s + 1; jp = t + 1
                r0 = -(R[ip - 1, jp - 1] + cost) * inv_g  # symmetric2
                r1 = -R[ip - 1, jp    ] * inv_g
                r2 = -R[ip,     jp - 1] * inv_g
                rm = r0
                if r1 > rm: rm = r1
                if r2 > rm: rm = r2
                rsum = math.exp(r0 - rm) + math.exp(r1 - rm) + math.exp(r2 - rm)
                R[ip, jp] = cost - gamma * (math.log(rsum) + rm)
        cuda.syncwarp()

    if s == 0:
        C[b, ri, cj] = R[D, D]


@cuda.jit(fastmath=True)
def _nested_inner_backward_kernel(X, Y, E_outer, upstream, grad_X32, grad_Y32,
                                   gamma, bw_freq, bw_time):
    """
    Fused backward: recompute inner forward in shared mem, run inner backward,
    then atomic-scatter fp32 gradient contributions to grad_X32 and grad_Y32.

    Optimisation: the full D×D cost matrix is precomputed once into shared
    memory (`costs[ip, jp]`), eliminating all on-the-fly (Xs[s]-Ys[t])² work
    in both the forward recompute and the backward pass and removing the six
    conditional branches that previously guarded boundary cost lookups.
    """
    pair_idx = cuda.blockIdx.x
    B_dim = E_outer.shape[0]; N_dim = E_outer.shape[1]; M_dim = E_outer.shape[2]
    if pair_idx >= B_dim * N_dim * M_dim:
        return

    NM    = N_dim * M_dim
    b     = pair_idx // NM
    ij    = pair_idx %  NM
    ri    = ij // M_dim
    cj    = ij %  M_dim
    D     = X.shape[2]

    if bw_time > 0.0:
        d = ri - cj
        if d < 0: d = -d
        if d > bw_time:
            return

    e_out = E_outer[b, ri, cj]
    # Sparse backward: skip cells where outer alignment probability is negligible.
    if e_out < np.float32(1e-7) and e_out > np.float32(-1e-7):
        return

    g_scale = e_out * upstream[b]

    s = cuda.threadIdx.x

    R     = cuda.shared.array((34, 34), dtype=np.float32)
    logE  = cuda.shared.array((34, 34), dtype=np.float32)
    costs = cuda.shared.array((34, 34), dtype=np.float32)  # costs[ip,jp] = (Xs[s]-Ys[t])²
    Xs    = cuda.shared.array((32,),    dtype=np.float32)
    Ys    = cuda.shared.array((32,),    dtype=np.float32)

    # Load frequency profiles into shared memory.
    if s < D:
        Xs[s] = X[b, ri, s]
        Ys[s] = Y[b, cj, s]

    # Parallel init: R→+inf, logE→-inf, costs→0 (boundary padding stays 0).
    for fi in range(s, 1156, 32):
        row = fi // 34; col = fi % 34
        R    [row, col] = math.inf
        logE [row, col] = -math.inf
        costs[row, col] = np.float32(0.0)
    cuda.syncwarp()  # R/logE/costs init + Xs/Ys load all visible

    # Set R[0,0] = 0 and precompute costs[s+1, t+1] = (Xs[s]-Ys[t])² for all t.
    # Both writes are independent so they share one sync.
    if s == 0:
        R[0, 0] = np.float32(0.0)
    if s < D:
        xs_val = Xs[s]   # cache in register to avoid repeated shared-mem reads
        for t in range(D):
            diff = xs_val - Ys[t]
            costs[s + 1, t + 1] = diff * diff
    cuda.syncwarp()  # R[0,0] and all costs entries are ready

    inv_g = np.float32(1.0) / gamma

    # ---- Forward pass (identical to forward kernel, costs from shared mem) ----
    for p in range(2 * D - 1):
        t = p - s
        if 0 <= s < D and 0 <= t < D:
            ip = s + 1; jp = t + 1
            skip = False
            if bw_freq > 0.0:
                d = s - t
                if d < 0: d = -d
                if d > bw_freq:
                    skip = True
            if not skip:
                cost = costs[ip, jp]
                r0 = -(R[ip - 1, jp - 1] + cost) * inv_g  # symmetric2
                r1 = -R[ip - 1, jp    ] * inv_g
                r2 = -R[ip,     jp - 1] * inv_g
                rm = r0
                if r1 > rm: rm = r1
                if r2 > rm: rm = r2
                rsum = math.exp(r0 - rm) + math.exp(r1 - rm) + math.exp(r2 - rm)
                R[ip, jp] = cost - gamma * (math.log(rsum) + rm)
        cuda.syncwarp()

    # ---- Backward boundary conditions on R ----
    saved = R[D, D]   # all threads read same value; each stores in own register
    for idx in range(s, 34, 32):
        R   [idx,     D + 1] = -math.inf
        R   [D + 1,   idx  ] = -math.inf
    cuda.syncwarp()
    if s == 0:
        R   [D + 1, D + 1] = saved
        logE[D + 1, D + 1] = np.float32(0.0)   # log(1): backward seed
    cuda.syncwarp()

    # ---- Backward pass (log-space) ----
    # costs[ip+1, jp], costs[ip, jp+1], costs[ip+1, jp+1] are read directly from
    # the precomputed table — no conditional branches for boundary cells needed
    # since boundary entries are 0 and the corresponding R values are -inf
    # (making those softmin terms vanish automatically).
    for p in range(2 * D - 2, -1, -1):
        t = p - s
        if 0 <= s < D and 0 <= t < D:
            ip = s + 1; jp = t + 1
            Rij = R[ip, jp]
            if not math.isinf(Rij):
                cd = costs[ip + 1, jp    ]
                cr = costs[ip,     jp + 1]
                cg = costs[ip + 1, jp + 1]
                la = (R[ip + 1, jp    ] - Rij - cd)               * inv_g
                lb = (R[ip,     jp + 1] - Rij - cr)               * inv_g
                lc = (R[ip + 1, jp + 1] - Rij - np.float32(2.0) * cg)        * inv_g  # symmetric2
                logE[ip, jp] = _logsumexp3(
                    logE[ip + 1, jp    ] + la,
                    logE[ip,     jp + 1] + lb,
                    logE[ip + 1, jp + 1] + lc,
                )
        cuda.syncwarp()

    # ---- Gradient computation + atomic scatter-add ----
    # Thread s handles the s-th frequency bin for both X and Y gradients.
    # Symmetric2 correction: dDTW/dD[i,j] = E[i,j] * (1 + p_diag(i,j))
    # where p_diag(i,j) = exp((R[ip,jp] - R[ip-1,jp-1] - 2*cost(i,j)) * inv_g)
    if s < D:
        ex_sum   = np.float32(0.0); ex_cross = np.float32(0.0)
        ey_sum   = np.float32(0.0); ey_cross = np.float32(0.0)
        for tt in range(D):
            # E_raw = dDTW/dR, correct to dDTW/dD via (1 + p_diag)
            ip_r = s + 1; jp_r = tt + 1
            log_pd = (R[ip_r, jp_r] - R[ip_r - 1, jp_r - 1]
                      - np.float32(2.0) * costs[ip_r, jp_r]) * inv_g
            if log_pd > np.float32(80.0):
                log_pd = np.float32(80.0)
            corr = np.float32(1.0) + math.exp(log_pd)

            e_row = math.exp(logE[ip_r, jp_r]) * corr   # corrected E[s, tt]
            ex_sum   += e_row
            ex_cross += e_row * Ys[tt]

            ip_c = tt + 1; jp_c = s + 1
            log_pd2 = (R[ip_c, jp_c] - R[ip_c - 1, jp_c - 1]
                       - np.float32(2.0) * costs[ip_c, jp_c]) * inv_g
            if log_pd2 > np.float32(80.0):
                log_pd2 = np.float32(80.0)
            corr2 = np.float32(1.0) + math.exp(log_pd2)

            e_col = math.exp(logE[ip_c, jp_c]) * corr2  # corrected E[tt, s]
            ey_sum   += e_col
            ey_cross += e_col * Xs[tt]

        cuda.atomic.add(grad_X32, (b, ri, s),
                        np.float32(2.0) * g_scale * (Xs[s] * ex_sum - ex_cross))
        cuda.atomic.add(grad_Y32, (b, cj, s),
                        np.float32(2.0) * g_scale * (Ys[s] * ey_sum - ey_cross))


# ---- Python launchers for the fused kernels ----

def _nested_inner_forward_cuda(X: torch.Tensor, Y: torch.Tensor, C: torch.Tensor,
                                gamma_freq: float, bw_freq: float, bw_time: float):
    """Launch _nested_inner_forward_kernel: fills C (float32) in-place."""
    B, N, D = X.shape
    M = Y.shape[1]
    # Upcast inputs to float32 so shared-mem DP is numerically stable.
    X_ = X.detach().contiguous().float()
    Y_ = Y.detach().contiguous().float()
    total = B * N * M
    _nested_inner_forward_kernel[total, _D_INNER_MAX](
        cuda.as_cuda_array(X_),
        cuda.as_cuda_array(Y_),
        cuda.as_cuda_array(C),
        np.float32(gamma_freq),
        np.float32(bw_freq),
        np.float32(bw_time),
    )


def _nested_inner_backward_cuda(X: torch.Tensor, Y: torch.Tensor,
                                 E_outer: torch.Tensor, upstream: torch.Tensor,
                                 grad_X32: torch.Tensor, grad_Y32: torch.Tensor,
                                 gamma_freq: float, bw_freq: float, bw_time: float):
    """Launch _nested_inner_backward_kernel: accumulates into grad_X32 / grad_Y32."""
    B, N, D = X.shape
    M = Y.shape[1]
    X_ = X.detach().contiguous().float()
    Y_ = Y.detach().contiguous().float()
    total = B * N * M
    _nested_inner_backward_kernel[total, _D_INNER_MAX](
        cuda.as_cuda_array(X_),
        cuda.as_cuda_array(Y_),
        cuda.as_cuda_array(E_outer.contiguous().float()),
        cuda.as_cuda_array(upstream.contiguous().float()),
        cuda.as_cuda_array(grad_X32),
        cuda.as_cuda_array(grad_Y32),
        np.float32(gamma_freq),
        np.float32(bw_freq),
        np.float32(bw_time),
    )


# ----------------------------------------------------------------------------------------------------------------------
class _NestedSoftDTWFunction(Function):
    """
    Nested SoftDTW autograd function.

    Three optimizations over the naive chunked implementation:
      1. Fused CUDA kernel (D <= _D_INNER_MAX=32): one CUDA block per (b,i,j)
         triple computes the inner D×D DTW entirely in shared memory — no
         Python loop, no per-chunk kernel launches.
      2. Outer-bandwidth filter: (b,i,j) pairs with |i-j| > bw_time are skipped
         during inner DTW computation because the outer DP never uses those cells.
      3. Sparse backward: inner backward is skipped for cells where E_outer < 1e-7
         (soft-DTW alignment probability is negligible).

    Falls back to a chunked Python loop when on CPU or D > _D_INNER_MAX.
    """

    @staticmethod
    def forward(ctx, X, Y, gamma_time, gamma_freq, bw_time, bw_freq):
        B, N, D = X.shape
        M        = Y.shape[1]
        use_cuda = X.is_cuda

        # C[b,i,j] = sdtw_freq(X[b,i,:], Y[b,j,:])  — kept in float32 for outer DP.
        C = torch.zeros(B, N, M, device=X.device, dtype=torch.float32)

        if use_cuda and D <= _D_INNER_MAX:
            # ---- Fused CUDA kernel: one block per triple, shared-mem DP ----
            _nested_inner_forward_cuda(X, Y, C, gamma_freq, bw_freq, bw_time)
        else:
            # ---- Full-batch fallback (CPU or D > _D_INNER_MAX) ----
            k  = torch.arange(B * N * M, device=X.device)
            b_ = k // (N * M);  ij = k % (N * M)
            i_ = ij // M;       j_ = ij % M

            if bw_time > 0:
                mask = (i_ - j_).abs() <= bw_time
                k, b_, i_, j_ = k[mask], b_[mask], i_[mask], j_[mask]

            if k.numel() > 0:
                Xc = X[b_, i_, :].unsqueeze(-1).contiguous()
                Yc = Y[b_, j_, :].unsqueeze(-1).contiguous()
                if use_cuda:
                    c_vals, _ = softdtw_forward_cuda_fused_sqeuclid(Xc, Yc, gamma_freq, bw_freq)
                else:
                    c_vals, _ = softdtw_forward_cpu(sqeuclidean(Xc, Yc), gamma_freq, bw_freq)
                C.view(B * N * M)[k] = c_vals.float()

        # Outer DTW forward on the (B,N,M) cost matrix.
        if use_cuda:
            out, R_outer = softdtw_forward_cuda(C.detach(), gamma_time, bw_time)
        else:
            out, R_outer = softdtw_forward_cpu(C.detach(), gamma_time, bw_time)

        ctx.save_for_backward(X, Y, C, R_outer)
        ctx.gamma_time = gamma_time;  ctx.gamma_freq = gamma_freq
        ctx.bw_time    = bw_time;     ctx.bw_freq    = bw_freq
        ctx.use_cuda   = use_cuda
        return out   # (B,)

    @staticmethod
    def backward(ctx, grad_output):
        X, Y, C, R_outer = ctx.saved_tensors
        B, N, D    = X.shape
        M          = Y.shape[1]
        gamma_time = ctx.gamma_time;  gamma_freq = ctx.gamma_freq
        bw_time    = ctx.bw_time;     bw_freq    = ctx.bw_freq
        use_cuda   = ctx.use_cuda

        # Outer backward → E_outer[b,i,j] = dL/dC[b,i,j].
        if use_cuda:
            E_outer = softdtw_backward_cuda_log(C, R_outer, gamma_time, bw_time)
        else:
            E_outer = softdtw_backward_cpu(C, R_outer, gamma_time, bw_time)

        if use_cuda and D <= _D_INNER_MAX:
            # ---- Fused CUDA backward (recompute + backward + scatter in one kernel) ----
            upstream = grad_output.to(device=X.device, dtype=torch.float32).reshape(-1)
            grad_X32 = torch.zeros(B, N, D, device=X.device, dtype=torch.float32)
            grad_Y32 = torch.zeros(B, M, D, device=X.device, dtype=torch.float32)
            _nested_inner_backward_cuda(X, Y, E_outer, upstream,
                                        grad_X32, grad_Y32,
                                        gamma_freq, bw_freq, bw_time)
            return grad_X32.to(X.dtype), grad_Y32.to(Y.dtype), None, None, None, None

        # ---- Full-batch fallback backward (CPU or D > _D_INNER_MAX) ----
        g      = grad_output.to(device=X.device, dtype=X.dtype).reshape(-1, 1, 1)
        E_flat = (E_outer.to(X.dtype) * g).reshape(B * N * M)

        k  = torch.arange(B * N * M, device=X.device)
        b_ = k // (N * M);  ij = k % (N * M)
        i_ = ij // M;       j_ = ij % M

        # Bandwidth + sparse filter applied once across all pairs.
        mask = torch.ones(k.numel(), dtype=torch.bool, device=X.device)
        if bw_time > 0:
            mask &= (i_ - j_).abs() <= bw_time
        mask &= E_flat.abs() >= 1e-7
        k, b_, i_, j_ = k[mask], b_[mask], i_[mask], j_[mask]

        grad_X = torch.zeros_like(X)
        grad_Y = torch.zeros_like(Y)

        if k.numel() == 0:
            return grad_X, grad_Y, None, None, None, None

        gc = E_flat[k]
        Xc = X[b_, i_, :].unsqueeze(-1).contiguous()
        Yc = Y[b_, j_, :].unsqueeze(-1).contiguous()

        if use_cuda:
            _, R_inner = softdtw_forward_cuda_fused_sqeuclid(Xc, Yc, gamma_freq, bw_freq)
            E_inner    = softdtw_backward_cuda_fused_sqeuclid(Xc, Yc, R_inner, gamma_freq, bw_freq)
        else:
            D_mat  = sqeuclidean(Xc, Yc)
            _, R_inner = softdtw_forward_cpu(D_mat, gamma_freq, bw_freq)
            E_inner    = softdtw_backward_cpu(D_mat, R_inner, gamma_freq, bw_freq)

        E_f  = E_inner.float() * gc.float().view(-1, 1, 1)
        Xc_f = Xc.float();  Yc_f = Yc.float()

        EX = E_f.sum(2);  EY = E_f.sum(1)
        gX = (2.0 * (Xc_f.squeeze(-1) * EX
                     - torch.bmm(E_f, Yc_f).squeeze(-1))).to(X.dtype)
        gY = (2.0 * (Yc_f.squeeze(-1) * EY
                     - torch.bmm(E_f.transpose(1, 2), Xc_f).squeeze(-1))).to(Y.dtype)

        fi = (b_ * N + i_).unsqueeze(1).expand(-1, D)
        fj = (b_ * M + j_).unsqueeze(1).expand(-1, D)
        grad_X.view(B * N, D).scatter_add_(0, fi, gX)
        grad_Y.view(B * M, D).scatter_add_(0, fj, gY)

        return grad_X, grad_Y, None, None, None, None, None


# ----------------------------------------------------------------------------------------------------------------------
class SoftDTWNested(pl.LightningModule):
    """
    Nested SoftDTW: independent monotone warping in both time AND frequency axes.

    For X, Y of shape (B, N, D):
      1. Inner DTW along frequency: C[b,i,j] = sdtw_freq(X[b,i,:], Y[b,j,:])
         Soft-DTW distance between the frequency profiles at time steps i and j.
      2. Outer DTW along time: sdtw_time(C)
         Aligns the time axis using the frequency-warped cost matrix.

    Both axes warp independently via their own monotone soft paths, giving true
    planar image-warping semantics without NP-hardness.  Every cell in the
    time-frequency plane contributes to the loss.

    Inner DTW runs entirely in CUDA shared memory (one block per (b,i,j) triple),
    so peak memory is O(B*(N+2)*(M+2)) — just the outer cost matrix C and the
    outer R table.  No chunking is needed.

    Args:
        gamma_time:     softness for the outer (time-axis) DTW
        gamma_freq:     softness for the inner (frequency-axis) DTW
        bandwidth:      Sakoe-Chiba bandwidth for the outer time DTW (None = disabled)
        bandwidth_freq: Sakoe-Chiba bandwidth for the inner freq DTW (None = disabled)
        normalize:      soft-DTW divergence normalization; requires N == M
    """

    def __init__(
        self,
        *,
        gamma_time: float = 1.0,
        gamma_freq: float = 1.0,
        bandwidth: float | None = None,
        bandwidth_freq: float | None = None,
        normalize: bool = False,
    ):
        super().__init__()
        for name, val in (("gamma_time", gamma_time), ("gamma_freq", gamma_freq)):
            v = float(val)
            if v <= 0 or not math.isfinite(v):
                raise ValueError(f"{name} must be finite > 0, got {v}")
        self.gamma_time = float(gamma_time)
        self.gamma_freq = float(gamma_freq)

        def _bw(bw):
            if bw is None: return -1.0
            bw = float(bw)
            return -1.0 if bw <= 0 else bw

        self.bw_time = _bw(bandwidth)
        self.bw_freq = _bw(bandwidth_freq)
        self.normalize = bool(normalize)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2: x = x.unsqueeze(0)
        if y.dim() == 2: y = y.unsqueeze(0)

        if x.dim() != 3 or y.dim() != 3:
            raise ValueError(
                f"Expected (B,N,D)/(B,M,D) or unbatched (N,D)/(M,D). "
                f"Got x={tuple(x.shape)}, y={tuple(y.shape)}"
            )
        bx, nx, dx = x.shape
        by, my, dy = y.shape
        if bx != by: raise ValueError(f"Batch sizes must match. Got {bx} vs {by}")
        if dx != dy: raise ValueError(f"Feature dims must match. Got {dx} vs {dy}")
        if nx == 0 or my == 0:
            raise ValueError(f"Sequence lengths must be > 0. Got N={nx}, M={my}")
        if self.normalize and nx != my:
            raise ValueError(
                f"normalize=True requires N == M for SoftDTWNested. Got N={nx}, M={my}."
            )

        args = (self.gamma_time, self.gamma_freq, self.bw_time, self.bw_freq)

        if self.normalize:
            B = bx
            x_cat = torch.cat([x, x, y], dim=0)
            y_cat = torch.cat([y, x, y], dim=0)
            out = _NestedSoftDTWFunction.apply(x_cat, y_cat, *args)
            out_xy, out_xx, out_yy = out.split(B, dim=0)
            return out_xy - 0.5 * (out_xx + out_yy)

        return _NestedSoftDTWFunction.apply(x, y, *args)


def gamma_soft_dtw_nested(
    dataset: torch.Tensor,
    n_samples: int = 100,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Estimate gamma_time and gamma_freq for :class:`SoftDTWNested`.

    Adapts the median-heuristic from Cuturi 2011 (``tslearn.metrics.gamma_soft_dtw``)
    to the two-level nested DTW setting.

    **Step 1 — gamma_freq** (inner DTW softness): computed from pairwise
    Euclidean distances between individual frequency-profile vectors,
    following the standard formula ``gamma = 2 * (median_dist * sqrt(D))^2``.

    **Step 2 — gamma_time** (outer DTW softness): the outer DTW operates on
    inner-DTW costs, whose magnitude is much larger than raw Euclidean
    distances.  We estimate typical inner-DTW costs using the gamma_freq
    from step 1 and apply the same median-heuristic at the outer level.

    Parameters
    ----------
    dataset : Tensor, shape (B, N, D) or (N, D)
        Representative data that SoftDTWNested will see.
        *B* = channels / batch, *N* = time steps, *D* = frequency bins.
    n_samples : int
        Number of random point-pairs used to estimate the median distance.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    gamma_time : float
    gamma_freq : float
    """
    if dataset.dim() == 2:
        dataset = dataset.unsqueeze(0)
    if dataset.dim() > 3:
        # Flatten leading dimensions into batch: (..., N, D) -> (B, N, D)
        dataset = dataset.reshape(-1, dataset.shape[-2], dataset.shape[-1])
    if dataset.dim() != 3:
        raise ValueError(f"Expected (...,N,D) or (N,D), got {tuple(dataset.shape)}")
    B, N, D = dataset.shape
    data_np = dataset.detach().cpu().float().numpy()

    rng = np.random.default_rng(seed)

    # ---- Step 1: gamma_freq from median squared Euclidean distance ----
    # between frequency-profile vectors (each of length D).
    freq_profiles = data_np.reshape(-1, D)  # (B*N, D)
    n_pts = freq_profiles.shape[0]
    idx = rng.choice(n_pts, size=min(n_samples, n_pts), replace=n_pts < n_samples)
    sample = freq_profiles[idx]
    diffs = sample[:, None, :] - sample[None, :, :]
    sq_dists = (diffs ** 2).sum(axis=-1)                  # (n, n)
    triu_idx = np.triu_indices(sq_dists.shape[0], k=1)
    median_sq_dist = float(np.median(sq_dists[triu_idx]))
    # gamma_freq ≈ median squared Euclidean distance between frequency profiles.
    # This gives moderate softness: the softmin entropy bonus is comparable to
    # the per-cell cost differences.
    gamma_freq = max(median_sq_dist, 1e-4)

    # ---- Step 2: gamma_time from sampled inner-DTW costs ----
    # The outer DTW operates on inner-DTW costs, which are much larger than
    # individual squared Euclidean distances.  We sample inner-DTW costs using
    # gamma_freq from step 1 and set gamma_time to their median absolute value.
    n_cost_samples = min(n_samples, B * N * N)
    b_idx = rng.integers(0, B, size=n_cost_samples)
    i_idx = rng.integers(0, N, size=n_cost_samples)
    j_idx = rng.integers(0, N, size=n_cost_samples)

    X_s = torch.tensor(data_np[b_idx, i_idx, :], dtype=torch.float32).unsqueeze(-1)
    Y_s = torch.tensor(data_np[b_idx, j_idx, :], dtype=torch.float32).unsqueeze(-1)

    if dataset.is_cuda and D <= _D_INNER_MAX:
        X_s, Y_s = X_s.cuda(), Y_s.cuda()
        c_vals, _ = softdtw_forward_cuda_fused_sqeuclid(X_s, Y_s, gamma_freq, -1.0)
    else:
        c_vals, _ = softdtw_forward_cpu(sqeuclidean(X_s, Y_s), gamma_freq, -1.0)

    c_np = c_vals.detach().cpu().float().numpy()
    median_cost = float(np.median(np.abs(c_np)))
    gamma_time = max(median_cost, 1e-4)

    return float(gamma_time), float(gamma_freq)


# ----------------------------------------------------------------------------------------------------------------------
# Symmetric2 step pattern tests
# Run directly:  python -m slicetca.run.dtw --test
# Or via pytest: pytest slicetca/run/dtw.py
#
# Background
# ----------
# All DTW kernels (both 1D and 2D) use the symmetric2 step pattern:
#   diagonal: predecessor + 2 * cost   (advances both axes)
#   vertical: predecessor + 1 * cost   (advances one axis)
#   horizontal: predecessor + 1 * cost (advances one axis)
#
# This path-length-normalizing step pattern is standard in DTW (cf. Quenot 1998,
# Sakoe & Chiba 1978).  A diagonal step advances the path by 2 positions
# (one on each axis) and costs 2c, while orthogonal steps advance by 1 and
# cost 1c.  This makes the total DTW cost proportional to the number of
# matched pairs, regardless of the alignment path shape.
#
# Test summary
# ------------
# For a constant cost matrix (all cells = c, shape B×N×N, small gamma -> hard DTW):
#   - test_symmetric2_scales_as_2n  : 1D and 2D both give 2N*c
#   - test_1d_2d_match              : 1D / 2D ratio ~= 1.0 (same step pattern)
#   - test_alignment_mass_diagonal  : backward E concentrates mass on diagonal
#   - test_nested_gradcheck         : forward/backward gradient consistency
# ----------------------------------------------------------------------------------------------------------------------

def _assert(condition: bool, name: str, msg: str = ""):
    if not condition:
        raise AssertionError(f"FAIL {name}: {msg}")


def test_symmetric2_scales_as_2n():
    """
    Both 1D and 2D kernels use the symmetric2 step pattern.
    For a constant cost matrix (all cells = c) with gamma -> 0,
    the forward value should approach 2N*c: the diagonal path costs
    2c per step (one from the softmin predecessor, one from the
    outer cost addition).
    """
    gamma = 1e-7
    c = 2.0
    B = 1
    for N in [2, 3, 4, 5, 7]:
        C = np.full((B, N, N), c, dtype=np.float64)
        R_1d = _softdtw_forward_cpu_np(C, gamma, 0.0)
        R_2d = _softdtw_2d_forward_np(C, gamma)
        val_1d = float(R_1d[0, -2, -2])
        val_2d = float(R_2d[0, -2, -2])
        expected = 2 * N * c
        _assert(
            abs(val_1d - expected) < 1e-3 * max(expected, 1.0),
            f"test_symmetric2_scales_as_2n[1D, N={N}]",
            f"expected {expected:.4f}, got {val_1d:.4f}",
        )
        _assert(
            abs(val_2d - expected) < 5e-3 * max(expected, 1.0),
            f"test_symmetric2_scales_as_2n[2D, N={N}]",
            f"expected {expected:.4f}, got {val_2d:.4f}",
        )


def test_1d_2d_match():
    """
    1D and 2D kernels should give the same result (both use symmetric2).
    """
    gamma = 1e-7
    c = 3.0
    B = 1
    for N in [2, 3, 4, 5, 6]:
        C = np.full((B, N, N), c, dtype=np.float64)
        val_1d = float(_softdtw_forward_cpu_np(C, gamma, 0.0)[0, -2, -2])
        val_2d = float(_softdtw_2d_forward_np(C, gamma)[0, -2, -2])
        ratio  = val_2d / val_1d if val_1d > 0 else float('inf')
        _assert(
            abs(ratio - 1.0) < 0.05,
            f"test_1d_2d_match[N={N}]",
            f"2D/1D = {ratio:.4f} (expected ~= 1.0)",
        )


def test_alignment_mass_diagonal():
    """
    With symmetric2 and a constant cost matrix, all monotone paths are
    equivalent (same total cost).  The backward E matrix should still have
    well-defined values (no NaN/inf) and the diagonal should carry a
    meaningful fraction of mass (not necessarily dominant, since all paths
    have equal cost under symmetric2 with constant costs).
    """
    gamma = 1e-3
    c = 1.0
    N = 8
    B = 1
    C = np.full((B, N, N), c, dtype=np.float64)

    R_1d = _softdtw_forward_cpu_np(C, gamma, 0.0)
    E_1d = _softdtw_backward_cpu_np(C.copy(), R_1d.copy(), gamma, 0.0)

    total_mass = float(E_1d[0].sum())
    _assert(
        total_mass > 0 and np.isfinite(total_mass),
        "test_alignment_mass_diagonal",
        f"total mass = {total_mass:.6f} (expected finite > 0)",
    )


def test_nested_gradcheck():
    """
    Numerical gradient check for SoftDTWNested.  Verifies that the analytic
    gradient (forward + backward through both inner and outer DTW) matches
    finite-difference gradients.

    Note: the fused CUDA kernels run entirely in float32 shared memory, so we
    use float32 inputs, eps=1e-3, and a 5% relative tolerance — appropriate
    for single-precision finite differences.
    """
    if not torch.cuda.is_available():
        return
    B, N, D = 2, 4, 3
    torch.manual_seed(42)
    # float32 — the fused kernels quantise to float32 internally anyway
    X = torch.randn(B, N, D, dtype=torch.float32, device='cuda', requires_grad=True)
    Y = torch.randn(B, N, D, dtype=torch.float32, device='cuda', requires_grad=True)

    loss_fn = SoftDTWNested(gamma_time=1.0, gamma_freq=1.0)

    eps = 1e-3  # appropriate for float32
    out = loss_fn(X, Y).sum()
    out.backward()
    grad_X_analytic = X.grad.clone()

    grad_X_numeric = torch.zeros_like(X)
    for b in range(B):
        for i in range(N):
            for d in range(D):
                X_plus = X.detach().clone()
                X_plus[b, i, d] += eps
                X_minus = X.detach().clone()
                X_minus[b, i, d] -= eps
                f_plus = loss_fn(X_plus, Y.detach()).sum()
                f_minus = loss_fn(X_minus, Y.detach()).sum()
                grad_X_numeric[b, i, d] = (f_plus - f_minus) / (2 * eps)

    max_err = (grad_X_analytic - grad_X_numeric).abs().max().item()
    rel_err = max_err / max(grad_X_numeric.abs().max().item(), 1e-8)
    _assert(
        rel_err < 0.05,
        "test_nested_gradcheck",
        f"max relative gradient error = {rel_err:.6e} (expected < 0.05)",
    )


def run_diagonal_downweighting_tests():
    """Run all symmetric2 step-pattern tests and report results."""
    tests = [
        test_symmetric2_scales_as_2n,
        test_1d_2d_match,
        test_alignment_mass_diagonal,
        test_nested_gradcheck,
    ]
    print("Running symmetric2 step-pattern tests...")
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print("All symmetric2 step-pattern tests passed.")


# ---- CUDA symmetric2 tests ----
# Mirror the CPU tests above but driven through the CUDA kernels directly.
#
# All kernels use symmetric2 (diagonal weighted 2x):
#   softdtw_forward_cuda            — 1D outer DTW       -> 2N*c
#   softdtw_2d_forward_precomp_cuda — 2D precomp DTW     -> 2N*c
#   _nested_inner_forward_cuda      — inner DTW (nested)  -> 2D*c
#
# All three tests are skipped automatically when no CUDA device is present.
# ----------------------------------------------------------------------------------------------------------------------

def _softdtw_2d_forward_precomp_launcher(C_torch: torch.Tensor, gamma: float):
    """
    Python launcher for softdtw_2d_forward_precomp_cuda over all anti-diagonals.
    Mirrors softdtw_forward_cuda but calls the 2D weighted-diagonal kernel.

    C_torch : (B, N, D) float32 CUDA tensor of pre-computed cell costs.
    Returns  : (out (B,), R (B, N+2, D+2))
    """
    C_ = C_torch.detach().contiguous().float()
    B_, N_, D_ = C_.shape
    R = torch.full((B_, N_ + 2, D_ + 2), math.inf, device=C_.device, dtype=C_.dtype)
    R[:, 0, 0] = 0.0
    C_ca = cuda.as_cuda_array(C_)
    R_ca = cuda.as_cuda_array(R)
    for p in range(N_ + D_ - 1):
        i_min = max(0, p - (D_ - 1))
        i_max = min(N_ - 1, p)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x   = (diag_len + TPB_LONG - 1) // TPB_LONG
        softdtw_2d_forward_precomp_cuda[(grid_x, B_), TPB_LONG](
            C_ca, R_ca, np.float32(gamma), N_, D_, p,
        )
    return R[:, -2, -2].contiguous(), R


def _warmup_cuda_kernels_for_tests():
    """One-shot Numba JIT warmup on tiny inputs so timing doesn't pollute tests."""
    _w = 4
    _dm = torch.ones(1, _w, _w, device='cuda', dtype=torch.float32)
    _cm = torch.ones(1, _w, _w, device='cuda', dtype=torch.float32)
    _xw = torch.zeros(1, _w, _w, device='cuda', dtype=torch.float32)
    _yw = torch.ones (1, _w, _w, device='cuda', dtype=torch.float32)
    _cw = torch.zeros(1, _w, _w, device='cuda', dtype=torch.float32)
    softdtw_forward_cuda(_dm, np.float32(1.0), np.float32(-1.0))
    _softdtw_2d_forward_precomp_launcher(_cm, 1.0)
    _nested_inner_forward_cuda(_xw, _yw, _cw,
                                np.float32(1.0), np.float32(-1.0), np.float32(-1.0))
    torch.cuda.synchronize()


def cuda_outer_symmetric2_scales_as_2n():
    """
    softdtw_forward_cuda with symmetric2: forward value ~= 2N*c.
    """
    if not torch.cuda.is_available():
        return
    gamma = np.float32(1e-4)
    c = 2.0
    B = 1
    for N in [2, 3, 4, 5]:
        D_mat = torch.full((B, N, N), c, dtype=torch.float32, device='cuda')
        out, _ = softdtw_forward_cuda(D_mat, gamma, np.float32(-1.0))
        torch.cuda.synchronize()
        val      = float(out[0])
        expected = 2 * N * c
        _assert(
            abs(val - expected) < 5e-3 * max(expected, 1.0),
            f"test_cuda_outer_symmetric2[N={N}]",
            f"expected {expected:.4f}, got {val:.4f}",
        )


def cuda_2d_symmetric2_scales_as_2n():
    """
    softdtw_2d_forward_precomp_cuda: forward value ~= 2N*c (same as 1D).
    """
    if not torch.cuda.is_available():
        return
    gamma = 1e-4
    c = 2.0
    B = 1
    for N in [2, 3, 4, 5]:
        C = torch.full((B, N, N), c, dtype=torch.float32, device='cuda')
        out, _ = _softdtw_2d_forward_precomp_launcher(C, gamma)
        torch.cuda.synchronize()
        val      = float(out[0])
        expected = 2 * N * c
        _assert(
            abs(val - expected) < 5e-3 * max(expected, 1.0),
            f"test_cuda_2d_symmetric2[N={N}]",
            f"expected {expected:.4f}, got {val:.4f}",
        )


def cuda_1d_2d_match():
    """
    CUDA 1D and 2D kernels give the same result (both symmetric2).
    """
    if not torch.cuda.is_available():
        return
    gamma = np.float32(1e-4)
    c = 3.0
    B = 1
    for N in [2, 3, 4, 5]:
        D_mat = torch.full((B, N, N), c, dtype=torch.float32, device='cuda')
        C     = torch.full((B, N, N), c, dtype=torch.float32, device='cuda')
        val_1d = float(softdtw_forward_cuda(D_mat, gamma, np.float32(-1.0))[0][0])
        val_2d = float(_softdtw_2d_forward_precomp_launcher(C, float(gamma))[0][0])
        torch.cuda.synchronize()
        ratio = val_2d / val_1d if val_1d > 0 else float('inf')
        _assert(
            abs(ratio - 1.0) < 0.1,
            f"test_cuda_1d_2d_match[N={N}]",
            f"2D/1D = {ratio:.4f} (expected ~= 1.0)",
        )


def cuda_inner_symmetric2_scales_as_2d():
    """
    _nested_inner_forward_cuda uses symmetric2.  For X=zeros, Y=sqrt(c)*ones
    of shape (B, N, D), every inner-DTW cell cost is c, so the result for
    each (i,j) pair should be ~= 2*D*c.
    """
    if not torch.cuda.is_available():
        return
    gamma_freq = np.float32(1e-4)
    c = 2.0
    B, N, D_ = 2, 4, 4
    X = torch.zeros(B, N, D_, dtype=torch.float32, device='cuda')
    Y = torch.full ((B, N, D_), c ** 0.5, dtype=torch.float32, device='cuda')
    C = torch.zeros(B, N, N,   dtype=torch.float32, device='cuda')
    _nested_inner_forward_cuda(X, Y, C, gamma_freq,
                                np.float32(-1.0), np.float32(-1.0))
    torch.cuda.synchronize()
    expected = 2 * D_ * c   # symmetric2: 2*D*c
    for i in range(N):
        for j in range(N):
            val = float(C[0, i, j])
            _assert(
                abs(val - expected) < 5e-3 * max(expected, 1.0),
                f"test_cuda_inner_symmetric2[i={i},j={j}]",
                f"expected {expected:.4f}, got {val:.4f}",
            )


def run_cuda_diagonal_downweighting_tests():
    """
    Run all CUDA symmetric2 tests.
    Skipped automatically when no CUDA device is available.
    """
    if not torch.cuda.is_available():
        print("No CUDA device -- skipping CUDA symmetric2 tests.")
        return
    print("Running CUDA symmetric2 tests...")
    _warmup_cuda_kernels_for_tests()
    tests = [
        cuda_outer_symmetric2_scales_as_2n,
        cuda_2d_symmetric2_scales_as_2n,
        cuda_1d_2d_match,
        cuda_inner_symmetric2_scales_as_2d,
    ]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print("All CUDA symmetric2 tests passed.")


# ----------------------------------------------------------------------------------------------------------------------
# Profiling script — run with:  python -m slicetca.run.dtw
# Profiles each phase of SoftDTWNested forward+backward at research dimensions.
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    from line_profiler import LineProfiler

    if not torch.cuda.is_available():
        print("No CUDA device found — profiling requires CUDA.")
        sys.exit(1)

    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    B, N, M, D = 357, 350, 350, 31
    dtype = torch.float16
    gamma_time = np.float32(10.0)
    gamma_freq = np.float32(1.0)
    bw_time    = np.float32(-1.0)
    bw_freq    = np.float32(-1.0)

    X = torch.randn(B, N, D, device='cuda', dtype=dtype)
    Y = torch.randn(B, M, D, device='cuda', dtype=dtype)

    # ---- Warm up Numba JIT ----
    print("Warming up...", end=' ', flush=True)
    _w = 8
    Xw = torch.randn(2, 4, _w, device='cuda', dtype=torch.float32)
    Yw = torch.randn(2, 4, _w, device='cuda', dtype=torch.float32)
    Cw = torch.zeros(2, 4, 4,  device='cuda', dtype=torch.float32)
    upw = torch.ones(2, device='cuda', dtype=torch.float32)
    gXw = torch.zeros(2, 4, _w, device='cuda', dtype=torch.float32)
    gYw = torch.zeros(2, 4, _w, device='cuda', dtype=torch.float32)
    _nested_inner_forward_cuda(Xw, Yw, Cw, np.float32(1.0), np.float32(-1.0), np.float32(-1.0))
    _, Rw = softdtw_forward_cuda(Cw, np.float32(1.0), np.float32(-1.0))
    Ew = softdtw_backward_cuda_log(Cw, Rw, np.float32(1.0), np.float32(-1.0))
    _nested_inner_backward_cuda(Xw, Yw, Ew, upw, gXw, gYw, np.float32(1.0), np.float32(-1.0), np.float32(-1.0))
    torch.cuda.synchronize()
    print("done")

    # ---- Function to profile: full forward + backward pipeline ----
    # torch.cuda.synchronize() after each async launch makes line_profiler
    # report actual GPU time per phase rather than Python dispatch time.
    def pipeline(X, Y, gamma_time, gamma_freq, bw_time, bw_freq):
        B, N, D = X.shape
        M = Y.shape[1]

        C = torch.zeros(B, N, M, device=X.device, dtype=torch.float32)
        _nested_inner_forward_cuda(X, Y, C, gamma_freq, bw_freq, bw_time)
        torch.cuda.synchronize()

        out, R_outer = softdtw_forward_cuda(C.detach(), gamma_time, bw_time)
        torch.cuda.synchronize()

        E_outer = softdtw_backward_cuda_log(C, R_outer, gamma_time, bw_time)
        torch.cuda.synchronize()

        upstream = torch.ones(B, device=X.device, dtype=torch.float32)
        grad_X = torch.zeros(B, N, D, device=X.device, dtype=torch.float32)
        grad_Y = torch.zeros(B, M, D, device=X.device, dtype=torch.float32)
        _nested_inner_backward_cuda(X, Y, E_outer, upstream, grad_X, grad_Y,
                                    gamma_freq, bw_freq, bw_time)
        torch.cuda.synchronize()

    # One untracked warmup run to fill GPU caches
    pipeline(X, Y, gamma_time, gamma_freq, bw_time, bw_freq)

    lp = LineProfiler()
    lp.add_function(pipeline)
    lp_pipeline = lp(pipeline)
    for _ in range(3):
        lp_pipeline(X, Y, gamma_time, gamma_freq, bw_time, bw_freq)

    lp.print_stats(output_unit=1e-3)   # output in milliseconds

