# Structure audit only. Does not assemble, factor, or call solve!.
using SDPA
for name in ("nsps_aug_nu1p0", "nsps_aug_nu0p92")
    p = SDPA.read_dat_s(joinpath(@__DIR__, name * ".dat-s"), Float64)
    plan = SDPA._certify_bordered(p)
    L = plan.layout
    @assert L.sizes == fill(2556, 4)
    @assert L.border == 35
    @assert L.order[end-34:end] == 1:35
    @assert L.slots == 13_429_854
    @assert L.inverse[L.order] == 1:p.m
    println(name, " rows=", p.m, " sizes=", L.sizes, " border=", L.border,
            " factor_panel_slots=", L.slots, " full_triangle=", SDPA._schur_triangle_slots(p.m))
end
