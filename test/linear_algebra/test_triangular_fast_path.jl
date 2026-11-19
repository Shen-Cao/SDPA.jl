# test/linear_algebra/test_triangular_fast_path.jl
#
# The solver's `mul!` calls with LowerTriangular/Adjoint wrappers
# (get_inv_chol_and_inv!, Lanczos matvecs, min_eigen QR paths) dispatch
# BigFloat to GenericLinearAlgebra's scalar path — slow and allocating.
# The fast path replaces them with explicit Rgemm!/mp_fma! kernels on
# the raw matrices.
#
# This file locks EQUIVALENCE of the reference wrapper path and the
# fast path at rtol 1e-30 (BigFloat 200-bit) / rtol 1e-15 (Float64).
# NOT bit-comparison: replacing a triangular matmul with a gemm-order
# loop legitimately changes accumulation order and rounding.

# Reference: min_eigen_lanczos with the reference (wrapper-mul!)
# kernels — a verbatim snapshot of the wrapper-mul! function body.
# Defined before the testset (top-level) so it is available at
# include time.
function _lanczos_old_kernels(
    l_mat::AbstractMatrix{T}, x_mat::AbstractMatrix{T}, max_iter::Int=0
) where {T<:Real}
    n = size(l_mat, 1)
    size(l_mat, 2) == n || error("l_mat must be square")
    size(x_mat) == (n, n) || error("x_mat must match l_mat size")

    iter_cap = max_iter > 0 ? min(max_iter, n) : min(n, floor(Int, sqrt(n)) + 10)

    q = SDPA._independent_zeros(T, n)
    q_old = SDPA._independent_zeros(T, n)
    r = SDPA._independent_zeros(T, n)
    SDPA._fill_independent!(r, one(T))
    beta = sqrt(T(n))

    alphas = SDPA._independent_zeros(T, n)
    betas = SDPA._independent_zeros(T, n)

    w = SDPA._independent_zeros(T, n)
    tmp = SDPA._independent_zeros(T, n)

    min_val = T(1e50)
    min_old = T(1e52)
    error = T(1e10)

    L_tri = LinearAlgebra.LowerTriangular(l_mat)

    kk = 0  # counter for periodic eigen-restart
    k = 0

    while k < iter_cap
        if beta <= T(1e-16)
            break
        end

        SDPA._copyto_independent!(q_old, q)
        inv_beta = one(T) / beta
        @inbounds @simd for i in 1:n
            q[i] = r[i] * inv_beta
        end

        # Reference kernels: wrapper mul!s.
        LinearAlgebra.mul!(w, LinearAlgebra.adjoint(L_tri), q)
        LinearAlgebra.mul!(tmp, x_mat, w)
        LinearAlgebra.mul!(w, L_tri, tmp)

        alpha = zero(T)
        @inbounds @simd for i in 1:n
            alpha += q[i] * w[i]
        end
        alphas[k + 1] = alpha

        @inbounds @simd for i in 1:n
            r[i] = w[i] - alpha * q[i] - beta * q_old[i]
        end

        if kk >= floor(Int, sqrt(max(1, k))) || k == n - 1 || k > SDPA.lanczos_restart_threshold(n)
            kk = 0
            kp1 = k + 1
            if kp1 >= 1
                diag = alphas[1:kp1]
                offdiag = kp1 >= 2 ? betas[1:kp1-1] : T[]
                if length(offdiag) == kp1 - 1
                    T_tri = LinearAlgebra.SymTridiagonal(diag, offdiag)
                    F = LinearAlgebra.eigen(T_tri)
                    min_val_new = F.values[1]
                    error = F.vectors[kp1, 1]
                    min_old = min_val
                    min_val = min_val_new
                end
            end
        end

        beta_sq = zero(T)
        @inbounds @simd for i in 1:n
            beta_sq += r[i] * r[i]
        end
        beta = sqrt(beta_sq)
        betas[k + 1] = beta

        if abs(min_val - min_old) <= T(1e-5) * abs(min_val) + T(1e-8) &&
           abs(error * beta) <= T(1e-2) * abs(min_val) + T(1e-4)
            break
        end

        k += 1
        kk += 1
    end

    return min_val - abs(error * beta)
end

@testset "triangular fast-path equivalence" begin
    # Deterministic fixtures (no RNG dependency): diagonal-dominated so
    # eigenvalue comparisons are well-conditioned at the gates below.
    # L: lower-triangular, diag = n, off-diag in [0, 0.45).
    # X: symmetric, diag = 1..n, off-diag in [0, 0.18).
    function fixtures(::Type{T}, n) where {T}
        L = zeros(T, n, n)
        X = zeros(T, n, n)
        for j in 1:n, i in 1:n
            if i == j
                L[i, j] = T(n)
                X[i, j] = T(i)
            elseif i > j
                L[i, j] = T(mod(i * 3 + j * 7, 10)) / T(20)
                X[i, j] = T(mod(i * 3 + j * 5, 10)) / T(50)
                X[j, i] = X[i, j]
            end
        end
        return L, X
    end

    setprecision(BigFloat, 200) do
        for T in (BigFloat, Float64)
            @testset "$T" begin
                rtol = T === BigFloat ? BigFloat("1e-30") : Float64(1e-15)

                # --- kernel: (L⁻¹)ᵀ·L⁻¹ (get_inv_chol_and_inv! product) ---
                @testset "kernel: (L')*L via Rgemm!(\"T\",\"N\")" begin
                    for n in (5, 15, 30)
                        M, _ = fixtures(T, n)  # any lower-tri M works for the kernel
                        ref = SDPA._independent_zeros(T, n, n)
                        fast = SDPA._independent_zeros(T, n, n)
                        # Reference kernel:
                        LinearAlgebra.mul!(
                            ref,
                            LinearAlgebra.adjoint(LinearAlgebra.LowerTriangular(M)),
                            LinearAlgebra.LowerTriangular(M),
                        )
                        # Fast kernel:
                        SDPA.Rgemm!("T", "N", n, n, n, one(T), M, M, zero(T), fast)
                        @test isapprox(fast, ref; rtol=rtol)
                    end
                end

                # --- kernel: Lanczos matvecs L'*q, X*w, L*tmp ---
                # (BigFloat only: the Float64 path keeps the wrapper mul! —
                # BLAS gemv is faster than a scalar loop for isbits types.)
                if T === BigFloat
                    @testset "kernel: Lanczos matvecs (mp_fma! loops)" begin
                        for n in (5, 15, 30)
                            L, X = fixtures(T, n)
                            q = SDPA._independent_zeros(T, n)
                            for i in 1:n
                                q[i] = T(mod(i * 7, 11) + 1) / T(3)
                            end
                            w_ref = SDPA._independent_zeros(T, n)
                            tmp_ref = SDPA._independent_zeros(T, n)
                            Lw = LinearAlgebra.LowerTriangular(L)
                            # Reference kernels:
                            LinearAlgebra.mul!(w_ref, LinearAlgebra.adjoint(Lw), q)
                            LinearAlgebra.mul!(tmp_ref, X, w_ref)
                            LinearAlgebra.mul!(w_ref, Lw, tmp_ref)
                            # Fast kernels:
                            w_fast = SDPA._independent_zeros(T, n)
                            tmp_fast = SDPA._independent_zeros(T, n)
                            for i in 1:n  # w = L' * q (L' upper-triangular)
                                SDPA.mp_zero!(w_fast[i])
                                for j in i:n
                                    SDPA.mp_fma!(w_fast[i], L[j, i], q[j], w_fast[i])
                                end
                            end
                            for i in 1:n  # tmp = X * w
                                SDPA.mp_zero!(tmp_fast[i])
                                for j in 1:n
                                    SDPA.mp_fma!(tmp_fast[i], X[i, j], w_fast[j], tmp_fast[i])
                                end
                            end
                            for i in 1:n  # w = L * tmp (L lower-triangular)
                                SDPA.mp_zero!(w_fast[i])
                                for j in 1:i
                                    SDPA.mp_fma!(w_fast[i], L[i, j], tmp_fast[j], w_fast[i])
                                end
                            end
                            @test isapprox(w_fast, w_ref; rtol=rtol)
                            @test isapprox(tmp_fast, tmp_ref; rtol=rtol)
                        end
                    end
                end

                # --- kernel: QR path products X*L' and L*(X*L') ---
                @testset "kernel: QR path (X*L', L*work) via Rgemm!" begin
                    for n in (5, 15, 30)
                        L, X = fixtures(T, n)
                        work_ref = SDPA._independent_zeros(T, n, n)
                        out_ref = SDPA._independent_zeros(T, n, n)
                        work_fast = SDPA._independent_zeros(T, n, n)
                        out_fast = SDPA._independent_zeros(T, n, n)
                        Lw = LinearAlgebra.LowerTriangular(L)
                        # Reference kernels:
                        LinearAlgebra.mul!(work_ref, X, LinearAlgebra.adjoint(Lw))
                        LinearAlgebra.mul!(out_ref, Lw, work_ref)
                        # Fast kernels:
                        SDPA.Rgemm!("N", "T", n, n, n, one(T), X, L, zero(T), work_fast)
                        SDPA.Rgemm!("N", "N", n, n, n, one(T), L, work_fast, zero(T), out_fast)
                        @test isapprox(work_fast, work_ref; rtol=rtol)
                        @test isapprox(out_fast, out_ref; rtol=rtol)
                    end
                end

                # --- e2e: get_inv_chol_and_inv! (serial + threaded) vs
                # reference-kernel reference on the example1 fixture ---
                @testset "e2e: get_inv_chol_and_inv! (serial + threaded)" begin
                    problem = SDPA.read_dat_s(
                        joinpath(@__DIR__, "..", "fixtures", "example1.dat-s"), T
                    )
                    params = SDPA.Parameters{T}()
                    current_pt = SDPA.SDPASolution{T}(
                        problem.m, problem.blockStruct, params.lambdaStar
                    )
                    a_mat = current_pt.zMat

                    if T === BigFloat
                        # Aliasing gate: setidentity! must give every
                        # diagonal slot its OWN BigFloat object — the
                        # in-place Rgemm!/mp_fma! product in
                        # get_inv_chol_and_inv! mutates the diagonal from
                        # many positions at once.
                        invz_blk = current_pt.invzMat.SDP_block[1].de_ele
                        @test invz_blk[1, 1] !== invz_blk[2, 2]
                        @test invz_blk[1, 1] !== invz_blk[1, 2]
                        z_blk = current_pt.zMat.SDP_block[1].de_ele
                        @test z_blk[1, 1] !== z_blk[2, 2]
                    end

                    # Reference: get_inv_chol! + reference
                    # wrapper-kernel product + symmetrize + LP division.
                    inv_chol_ref = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    inv_ref = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    work_mat_ref = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    @test SDPA.get_inv_chol!(inv_chol_ref, a_mat, work_mat_ref)
                    L_inv = inv_chol_ref.SDP_block[1].de_ele
                    inv_ref_blk = inv_ref.SDP_block[1].de_ele
                    n = size(L_inv, 1)
                    # Reference kernel:
                    LinearAlgebra.mul!(
                        inv_ref_blk,
                        LinearAlgebra.adjoint(LinearAlgebra.LowerTriangular(L_inv)),
                        LinearAlgebra.LowerTriangular(L_inv),
                    )
                    for j in 1:n, i in 1:j
                        avg = (inv_ref_blk[i, j] + inv_ref_blk[j, i]) / 2
                        inv_ref_blk[i, j] = avg
                        inv_ref_blk[j, i] = avg
                    end
                    @inbounds for k in eachindex(a_mat.LP_block)
                        inv_ref.LP_block[k] = one(T) / a_mat.LP_block[k]
                    end

                    # Fixed serial path:
                    inv_chol_fast = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    inv_fast = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    work_mat_fast = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    @test SDPA.get_inv_chol_and_inv!(
                        inv_chol_fast, inv_fast, a_mat, work_mat_fast
                    )
                    @test isapprox(inv_fast.SDP_block[1].de_ele, inv_ref.SDP_block[1].de_ele; rtol=rtol)
                    @test inv_fast.LP_block == inv_ref.LP_block  # unchanged code path — exact

                    # Fixed threaded path (same kernel as serial — the
                    # cross-thread-count bit-identity gate):
                    inv_chol_thr = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    inv_thr = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    work_mat_thr = SDPA.SDPADenseLinearSpace{T}(problem.blockStruct)
                    @test SDPA.threaded_get_inv_chol_and_inv!(
                        inv_chol_thr, inv_thr, a_mat, work_mat_thr
                    )
                    @test isapprox(inv_thr.SDP_block[1].de_ele, inv_ref.SDP_block[1].de_ele; rtol=rtol)
                end

                # --- e2e: min_eigen_dispatch QR path (n ≤ 20) vs
                # reference-kernel reference ---
                # Float64 uses rtol 1e-12 here: the eigensolve amplifies
                # kernel-level ulp differences by the (small, O(1)) condition
                # number of the eigenproblem, which can exceed 1e-15. The
                # 1e-15 Float64 gate is enforced at the kernel level above.
                @testset "e2e: min_eigen_dispatch QR path" begin
                    e2e_rtol = T === BigFloat ? BigFloat("1e-30") : Float64(1e-12)
                    for n in (5, 15)
                        L, X = fixtures(T, n)
                        work_ref = SDPA._independent_zeros(T, n, n)
                        out_ref = SDPA._independent_zeros(T, n, n)
                        Lw = LinearAlgebra.LowerTriangular(L)
                        # Reference kernels:
                        LinearAlgebra.mul!(work_ref, X, LinearAlgebra.adjoint(Lw))
                        LinearAlgebra.mul!(out_ref, Lw, work_ref)
                        lam_ref = SDPA.min_eigen_qr(out_ref)
                        lam_new = SDPA.min_eigen_dispatch(L, X)
                        @test isapprox(lam_new, lam_ref; rtol=e2e_rtol)
                    end
                end

                # --- e2e: min_eigen_lanczos vs reference ---
                # The reference is the wrapper-mul! function body.
                # Both trajectories are deterministic; with kernels
                # differing only at the ~1e-60 level the iteration counts
                # coincide, so rtol 1e-30 is reachable.
                if T === BigFloat
                    @testset "e2e: min_eigen_lanczos" begin
                        for n in (15, 30)
                            L, X = fixtures(T, n)
                            lam_ref = _lanczos_old_kernels(L, X)
                            lam_new = SDPA.min_eigen_lanczos(L, X)
                            @test isapprox(lam_new, lam_ref; rtol=BigFloat("1e-30"))
                        end
                    end
                end
            end
        end
    end
end
