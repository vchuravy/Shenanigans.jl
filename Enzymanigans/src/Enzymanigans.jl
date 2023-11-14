module Enzymanigans

import EnzymeCore
import Enzyme

include("codecache.jl")
const GLOBAL_CI_CACHE = CodeCache()

import Core: MethodMatch, MethodTable
import Core.Compiler: _methods_by_ftype, InferenceParams, get_world_counter, MethodInstance,
    specialize_method, InferenceResult, typeinf, InferenceState, NativeInterpreter,
    code_cache, AbstractInterpreter, OptimizationParams, WorldView, MethodTableView, ArgInfo,
    StmtInfo, singleton_type, argtypes_to_type, MethodMatchInfo, CallMeta, widenconst, anymap

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

primal(c::EnzymeCore.Const) = c.val
primal(c::EnzymeCore.Duplicated) = c.val

function unwrap_annotation(@nospecialize(c))
    if c isa Core.Compiler.Const
        v = c.val
        if v isa Type && v <: EnzymeCore.Annotation
            return Core.Compiler.Const(eltype(v))
        end
        if v isa EnzymeCore.Annotation 
            return Core.Compiler.Const(primal(v))
        end
    elseif widenconst(c) != c
        error("Unhandled $c")
    elseif c <: Type{<:EnzymeCore.Annotation}
        c = c.parameters[1]
        return Type{eltype(c)}
    else c <: EnzymeCore.Annotation 
        return eltype(c)
    end
    return c
end

function dummy end
    
function Core.Compiler.abstract_call_gf_by_type(interp::EnzymeInterpreter, @nospecialize(f),
        arginfo::ArgInfo, si::StmtInfo, @nospecialize(atype), sv::InferenceState, max_methods::Int)
    (;argtypes) = arginfo

    if f === EnzymeCore.autodiff && length(argtypes) >= 4
        mode = argtypes[2]
        ft_activity = argtypes[3]
        maybe_rt_activity = argtypes[4]
        @info "Found EnzymeCore.autodiff" mode ft_activity maybe_rt_activity
        if widenconst(ft_activity)       <: EnzymeCore.Annotation &&
           widenconst(maybe_rt_activity) <: Type{<:EnzymeCore.Annotation}
            if widenconst(mode) <: EnzymeCore.ForwardMode
                @info "Found EnzymeCore.autodiff with full information"
                rt_activity = widenconst(maybe_rt_activity)
                inner_argtypes = argtypes[5:end] # strip rt_activity from primal
                pushfirst!(inner_argtypes, argtypes[3]) # push func
                primal_argtypes = anymap(unwrap_annotation, inner_argtypes)

                # 1. Infer primal
                ft = primal_argtypes[1]
                f = singleton_type(ft)
                call = Core.Compiler.abstract_call_gf_by_type(interp.parent,
                    f, ArgInfo(nothing, primal_argtypes), StmtInfo(true), argtypes_to_type(primal_argtypes), sv, max_methods)

                @show call
                # 2. Obtain MI
                info = call.info
                if call.info isa Core.Compiler.ConstCallInfo
                    info = info.call
                end
                if info isa MethodMatchInfo
                    if length(info.results.matches) == 0
                        @error "Found no matching call" primal_argtypes
                        error("Enzyme compilation failed")
                    end
                    mi = specialize_method(only(info.results.matches), preexisting=true)
                else
                    @error "Unknown" info call
                    error("")
                end

                # TODO: This is clunky
                width = Enzyme.same_or_one(anymap(widenconst, inner_argtypes)...)
                mode = Enzyme.API.DEM_ForwardMode

                rt_activity = only(rt_activity.parameters)
                ReturnPrimal = rt_activity <: EnzymeCore.Duplicated || rt_activity <: EnzymeCore.BatchDuplicated
                ModifiedBetween = ntuple(_->false, length(primal_argtypes))
                ShadowInit = false

                # 4. Compile
                params = Enzyme.Compiler.EnzymeCompilerParams(
                    argtypes_to_type(inner_argtypes),
                    mode, width, rt_activity, #=run_enzyme=# true,  #=abiwrap=#true,
                    ModifiedBetween, ReturnPrimal, ShadowInit, Enzyme.Compiler.UnknownTapeType, Enzyme.Compiler.FFIABI)
                @info "Configuration..." params rt_activity
            
                job = Enzyme.Compiler.CompilerJob(mi, Enzyme.Compiler.CompilerConfig(Enzyme.Compiler.EnzymeTarget(), params; kernel=false), interp.world)
                compile_result = Enzyme.Compiler.JuliaContext() do ctx # TODO: ctx should be sunk
                    Enzyme.Compiler.cached_compilation(job) # TODO this takes a lock. Safe from here?
                end
                FMT = Enzyme.Compiler.ForwardModeThunk{typeof(compile_result.adjoint), widenconst(ft_activity), rt_activity, Tuple{params.TT.parameters[2:end]...}, Val{width}, Val(ReturnPrimal)}

                args = [Symbol("#self#")]
                for i in 1:length(primal_argtypes)
                    push!(args, Symbol("arg_$i"))
                end
                
                # 3. Create an OpaqueClosure that contains the thunk statically and the enzymecall
                expr = Expr(:lambda, args,
                    Expr(Symbol("scope-block"), quote
                        thunk = $(FMT(compile_result.adjoint))
                        $(Expr(:call, :thunk, args[2:end]...))
                end))
                CI = @ccall jl_expand_and_resolve(expr::Any, Enzyme::Any, Core.Compiler.svec()::Core.SimpleVector)::Any
                # infer CI (see https://github.com/JuliaLang/julia/blob/9729f312186c70de4f5209dfa0d0fdf0f1669ee6/stdlib/REPL/src/REPLCompletions.jl#L631)

                dummy_mi = @ccall jl_new_method_instance_uninit()::Ref{Core.MethodInstance}
                @show dummy_mi.specTypes = Core.Compiler.signature_type(dummy, argtypes_to_type(inner_argtypes))

                dummy_mi.def = Enzyme
                @atomic dummy_mi.uninferred = CI

                @show result = Core.Compiler.InferenceResult(dummy_mi, Core.Compiler.SimpleArgtypes(Any[typeof(dummy), inner_argtypes...]))
                @show result.argtypes
                frame = Core.Compiler.InferenceState(result, CI, #=cache=#:no, interp.parent)
                Core.Compiler.typeinf(interp.parent, frame)

                @show frame.src
                @show oc = Core.OpaqueClosure(frame.src)

                # TODO: Cache
                # ci = get(rinterp.unopt[rinterp.current_level], mi, nothing)
                # clos = AbstractCompClosure(rinterp.current_level, 1, call.info, ci.stmt_info)
                # TODO: Construct a OpaqueClosure that contains the abicall
                # clos = Core.PartialOpaque(Core.OpaqueClosure{<:Tuple, <:Any}, nothing, sv.linfo, clos)

                # TODO: Based on rt_activity and call.rt compute the return type of the OpaqueClosure
                # return CallMeta(rt2, call.effects, call.info)
            else
                @error "Unkown mode" mode
            end

            # call = Core.Compiler.abstract_call_gf_by_type(interp.parent,
            #         f, ArgInfo(nothing, inner_argtypes), StmtInfo(true), argtypes_to_type(inner_argtypes), sv, max_methods)

            # # Now call Enzyme to generate OpaqueClosure
            # if isa(call.info, MethodMatchInfo)
            #     if length(call.info.results.matches) == 0
            #         @show inner_argtypes
            #         error("")
            #     end
            #     mi = specialize_method(call.info.results.matches[1], preexisting=true)
            #     # TODO: Cache
            #     # ci = get(rinterp.unopt[rinterp.current_level], mi, nothing)
            #     # clos = AbstractCompClosure(rinterp.current_level, 1, call.info, ci.stmt_info)
            #     # TODO: Construct a OpaqueClosure that contains the abicall
            #     # clos = Core.PartialOpaque(Core.OpaqueClosure{<:Tuple, <:Any}, nothing, sv.linfo, clos)
            # else
            #     @error "Unknown" inner_argtypes call.info call
            #     error("")
            # end
            # # TODO Obtain rt2
            # # rt2 = call.rt
            # rt2 = Float64
            # return CallMeta(rt2, call.effects, call.info)
        end
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
