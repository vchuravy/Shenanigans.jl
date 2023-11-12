module Enzymanigans

function autodiff(f, args...)
    f(args...)
end


include("codecache.jl")
const GLOBAL_CI_CACHE = CodeCache()

import Core: MethodMatch, MethodTable
import Core.Compiler: _methods_by_ftype, InferenceParams, get_world_counter, MethodInstance,
    specialize_method, InferenceResult, typeinf, InferenceState, NativeInterpreter,
    code_cache, AbstractInterpreter, OptimizationParams, WorldView, MethodTableView, ArgInfo,
    StmtInfo, singleton_type, argtypes_to_type, MethodMatchInfo, CallMeta

struct EnzymeInterpreter <: AbstractInterpreter
    global_cache::CodeCache
    method_table::Union{Nothing,Core.MethodTable}

    # Cache of inference results for this particular interpreter
    local_cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    parent::AbstractInterpreter


    function EnzymeInterpreter(cache::CodeCache, mt::Union{Nothing,Core.MethodTable}, world::UInt, ip::InferenceParams, op::OptimizationParams)
        @assert world <= Base.get_world_counter()

        return new(
            cache,
            mt,

            # Initially empty cache
            Vector{InferenceResult}(),

            # world age counter
            world,

            # parameters for inference and optimization
            ip,
            op,

            Core.Compiler.NativeInterpreter(world; inf_params=ip, opt_params=op)
        )
    end
end

Core.Compiler.InferenceParams(interp::EnzymeInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::EnzymeInterpreter) = interp.opt_params
Core.Compiler.get_world_counter(interp::EnzymeInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::EnzymeInterpreter) = interp.local_cache
Core.Compiler.code_cache(interp::EnzymeInterpreter) = WorldView(interp.global_cache, interp.world)

# No need to do any locking since we're not putting our results into the runtime cache
Core.Compiler.lock_mi_inference(interp::EnzymeInterpreter, mi::MethodInstance) = nothing
Core.Compiler.unlock_mi_inference(interp::EnzymeInterpreter, mi::MethodInstance) = nothing

Core.Compiler.may_optimize(interp::EnzymeInterpreter) = true
Core.Compiler.may_compress(interp::EnzymeInterpreter) = true
Core.Compiler.may_discard_trees(interp::EnzymeInterpreter) = false
Core.Compiler.verbose_stmt_info(interp::EnzymeInterpreter) = false

using Core.Compiler: OverlayMethodTable
Core.Compiler.method_table(interp::EnzymeInterpreter) =
    OverlayMethodTable(interp.world, interp.method_table)

Base.Experimental.@MethodTable(GLOBAL_METHOD_TABLE)

##
# MethodTable extensions 

function delete!(mt::Core.MethodTable, m::Method)
    ccall(:jl_method_table_disable, Cvoid, (Any, Any), mt, m)
end

function whichtt(mt::MethodTable, sig)
    mtv = OverlayMethodTable(Base.get_world_counter(), mt)
    whichtt(mtv, sig)
end

function whichtt(mtv::MethodTableView, sig)
    match, valid_worlds, overlayed = Core.Compiler.findsup(sig, mtv)
    match === nothing && return nothing
    return match.method
end

function Core.Compiler.abstract_call_gf_by_type(interp::EnzymeInterpreter, @nospecialize(f),
        arginfo::ArgInfo, si::StmtInfo, @nospecialize(atype), sv::InferenceState, max_methods::Int)
    (;argtypes) = arginfo

    if f === autodiff
        # Use Parent to infer rt
        inner_argtypes = argtypes[2:end]
        ft = inner_argtypes[1]
        f = singleton_type(ft)

        call = Core.Compiler.abstract_call_gf_by_type(interp.parent,
                f, ArgInfo(nothing, inner_argtypes), StmtInfo(true), argtypes_to_type(inner_argtypes), sv, max_methods)

        # Now call Enzyme to generate OpaqueClosure
        if isa(call.info, MethodMatchInfo)
            if length(call.info.results.matches) == 0
                @show inner_argtypes
                error("")
            end
            mi = specialize_method(call.info.results.matches[1], preexisting=true)
            # TODO: Cache
            # ci = get(rinterp.unopt[rinterp.current_level], mi, nothing)
            # clos = AbstractCompClosure(rinterp.current_level, 1, call.info, ci.stmt_info)
            # TODO: Construct a OpaqueClosure that contains the abicall
            # clos = Core.PartialOpaque(Core.OpaqueClosure{<:Tuple, <:Any}, nothing, sv.linfo, clos)
        else
            @error "Unknown" inner_argtypes call.info call
            error("")
        end
        # TODO Obtain rt2
        # rt2 = call.rt
        rt2 = Float64
        return CallMeta(rt2, call.effects, call.info)
    end

    ret = @invoke Core.Compiler.abstract_call_gf_by_type(interp::AbstractInterpreter, f::Any,
        arginfo::ArgInfo, si::StmtInfo, atype::Any, sv::InferenceState, max_methods::Int)

    return ret
end
#
##

##
# Future: CompilerPlugins.jl

function custom_invoke(f, args...)
    @nospecialize f args
    interp = EnzymeInterpreter(GLOBAL_CI_CACHE, GLOBAL_METHOD_TABLE, get_world_counter(), InferenceParams(), OptimizationParams())
    oc = construct_oc_in_absint(f, interp, args...)
    oc(args...)
end

function construct_oc_in_absint(f, interp, args...)
    @nospecialize f args
    tt = Base.signature_type(f, Tuple{map(Core.Typeof, args)...})
    match, valid_worlds, overlayed = Core.Compiler.findsup(tt, Core.Compiler.method_table(interp))
    if match === nothing
        error(lazy"Unable to find matching $tt")
    end
    mi = specialize_method(match.method, match.spec_types, match.sparams)::MethodInstance
    code = Core.Compiler.get(code_cache(interp), mi, nothing)
    if code !== nothing
        inf = code.inferred#::Vector{UInt8}
        ci = Base._uncompressed_ir(code, inf)
        return Core.OpaqueClosure(ci) # TODO cache
    end
    result = InferenceResult(mi)
    frame = InferenceState(result, #=cache=# :global, interp)
    typeinf(interp, frame)
    ci = frame.src
    return Core.OpaqueClosure(ci) # TODO cache
end

function get_single_method_match(@nospecialize(tt), lim, world)
    mms = _methods_by_ftype(tt, lim, world)
    isa(mms, Bool) && single_match_error(tt)
    local mm = nothing
    for i = 1:length(mms)
        mmᵢ = mms[i]::MethodMatch
        if tt === mmᵢ.spec_types
            mm === nothing || single_match_error(tt)
            mm = mmᵢ
        end
    end
    mm isa MethodMatch || single_match_error(tt)
    return mm
end

end # module Enzymanigans