# benchmark/diagnose_fill.jl
#
# Diagnose _fill_independent! precision behavior inside `setprecision`
# blocks: what precision do the freshly created elements get, how does it
# relate to the enclosing scope, and can a precision mismatch between a
# container and its elements corrupt in-place MPFR updates?

using SDPA
using Printf

println("=" ^ 60)
println("Diagnosing _fill_independent! precision behavior")
println("=" ^ 60)

# Test 1: Does _independent_zeros create truly independent elements?
println("\n--- Test 1: _independent_zeros independence ---")
setprecision(BigFloat, 200) do
    M = SDPA._independent_zeros(BigFloat, 3, 3)
    println("precision(M[1,1]) = ", precision(M[1,1]))
    println("precision(M[2,2]) = ", precision(M[2,2]))
    # Mutate [1,1] and check [2,2]
    SDPA.mp_fma!(M[1,1], BigFloat(2), BigFloat(3), M[1,1])
    println("After mp_fma!(M[1,1], 2, 3, M[1,1]):")
    println("  M[1,1] = ", M[1,1], " (expect 6)")
    println("  M[2,2] = ", M[2,2], " (expect 0)")
    @assert M[2,2] == 0  "ALIASING BUG: M[2,2] corrupted!"
    println("  PASS: elements are independent")
end

# Test 2: Does _fill_independent! create independent elements?
println("\n--- Test 2: _fill_independent! independence ---")
setprecision(BigFloat, 200) do
    M = zeros(BigFloat, 3, 3)  # shared elements initially
    println("Before _fill_independent!:")
    println("  precision(M[1,1]) = ", precision(M[1,1]))
    SDPA._fill_independent!(M, zero(BigFloat))
    println("After _fill_independent!:")
    println("  precision(M[1,1]) = ", precision(M[1,1]))
    println("  precision(M[2,2]) = ", precision(M[2,2]))
    SDPA.mp_fma!(M[1,1], BigFloat(2), BigFloat(3), M[1,1])
    println("After mp_fma!(M[1,1], 2, 3, M[1,1]):")
    println("  M[1,1] = ", M[1,1], " (expect 6)")
    println("  M[2,2] = ", M[2,2], " (expect 0)")
    if M[2,2] != 0
        println("  ALIASING BUG: M[2,2] = ", M[2,2])
    else
        println("  PASS: elements are independent")
    end
end

# Test 3: What precision do _fill_independent! elements get?
println("\n--- Test 3: _fill_independent! precision inside setprecision(200) ---")
setprecision(BigFloat, 200) do
    println("Global precision inside block: ", precision(BigFloat))
    M = Matrix{BigFloat}(undef, 3, 3)
    SDPA._fill_independent!(M, zero(BigFloat))
    for i in eachindex(M)
        println("  M[$i] precision = ", precision(M[i]))
    end
    # Now do an operation that should be 200-bit
    x = BigFloat("1.23456789")  # created inside block → 200-bit
    println("  x = BigFloat(\"1.23456789\") precision = ", precision(x))
    SDPA.mp_mul!(M[1,1], x, x)
    println("  After mp_mul!(M[1,1], x, x): M[1,1] = ", M[1,1])
    println("  M[1,1] precision = ", precision(M[1,1]))
end

# Test 4: Does BigFloat(0) inside _fill_independent! get scoped precision?
println("\n--- Test 4: BigFloat(0) precision at different scopes ---")
println("Default precision: ", precision(BigFloat))
setprecision(BigFloat, 200) do
    x1 = BigFloat(0)
    println("BigFloat(0) inside setprecision(200): precision = ", precision(x1))
    x2 = BigFloat(0, 200)
    println("BigFloat(0, 200): precision = ", precision(x2))
    x3 = zero(BigFloat)
    println("zero(BigFloat) inside setprecision(200): precision = ", precision(x3))
end

# Test 5: Full simulation — what happens if bmat is _fill_independent!'d
# inside a setprecision block, then mp_fma! is called?
println("\n--- Test 5: Simulated bmat fill + mp_fma! at 200-bit ---")
setprecision(BigFloat, 200) do
    # Simulate compute_bmat_dense! zeroing bmat
    bmat = zeros(BigFloat, 4, 4)
    println("Before fill: precision(bmat[1,1]) = ", precision(bmat[1,1]))
    SDPA._fill_independent!(bmat, zero(BigFloat))
    println("After _fill_independent!: precision(bmat[1,1]) = ", precision(bmat[1,1]))

    # Now simulate the F1 formula accumulation
    temp = BigFloat("1.5")  # 200-bit
    G11 = BigFloat("2.5")  # 200-bit
    println("temp precision = ", precision(temp))
    println("G11 precision = ", precision(G11))

    # F1: bmat[i,j] += temp * G[i,i] * G[j,j]
    SDPA.mp_mul!(temp, temp, G11)
    SDPA.mp_fma!(bmat[1,1], temp, G11, bmat[1,1])
    println("After mp_fma!(bmat[1,1], temp, G11, bmat[1,1]):")
    println("  bmat[1,1] = ", bmat[1,1])
    println("  bmat[1,1] precision = ", precision(bmat[1,1]))
    println("  bmat[2,2] = ", bmat[2,2], " (expect 0)")
    if bmat[2,2] != 0
        println("  ALIASING BUG DETECTED!")
    else
        println("  PASS: no aliasing, precision correct")
    end
end

# Test 6: The critical test — what precision does a bmat created OUTSIDE
# the setprecision block have, vs one created inside?
println("\n--- Test 6: bmat created outside vs inside setprecision ---")
# bmat_outside is created at default precision (256)
bmat_outside = SDPA._independent_zeros(BigFloat, 4, 4)
println("bmat_outside created at default precision:")
println("  precision(bmat_outside[1,1]) = ", precision(bmat_outside[1,1]))

setprecision(BigFloat, 200) do
    println("Inside setprecision(200):")
    println("  precision(bmat_outside[1,1]) = ", precision(bmat_outside[1,1]))
end

# Correct test:
setprecision(BigFloat, 200) do
    println("Inside setprecision(200) again:")
    println("  precision(bmat_outside[1,1]) = ", precision(bmat_outside[1,1]))
    SDPA._fill_independent!(bmat_outside, zero(BigFloat))
    println("After _fill_independent! inside block:")
    println("  precision(bmat_outside[1,1]) = ", precision(bmat_outside[1,1]))
end

println("\n" * "=" ^ 60)
println("Diagnosis complete.")
