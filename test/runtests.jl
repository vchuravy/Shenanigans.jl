using Test

include("stackedmethodtable.jl")

f_ambig(::Int, _) = 1
f_ambig(_, ::Int) = 2

Core.Compiler.findsup(Tuple{typeof(f_ambig), Int, Int}, Core.Compiler.InternalMethodTable(Base.get_world_counter()))

Base.Experimental.@MethodTable(UnionMT)
Base.Experimental.@MethodTable(AParentMT)
Base.Experimental.@MethodTable(BParentMT)

import Shenanigans: UnionMethodTable
UnionTable() = UnionMethodTable(Base.get_world_counter(), UnionMT, AParentMT, BParentMT)

@testset "Unoverlayed" begin
    @show u_sin = findsup(Tuple{typeof(sin), Float64}, UnionTable())
end