module Shenanigans

import Core: MethodTable
import Core.Compiler: MethodTableView, InternalMethodTable,
                        MethodMatchResult, MethodLookupResult, WorldRange
struct StackedMethodTable{MTV<:MethodTableView} <: MethodTableView
    world::UInt
    mt::MethodTable
    parent::MTV
end
StackedMethodTable(world::UInt, mt::MethodTable) = StackedMethodTable(world, mt, InternalMethodTable(world))
StackedMethodTable(world::UInt, mt::MethodTable, parent::MethodTable) = StackedMethodTable(world, mt, StackedMethodTable(world, parent))

import Core.Compiler: findall, _findall, length, vcat, isempty, max, min, getindex
function findall(@nospecialize(sig::Type), table::StackedMethodTable; limit::Int=-1)
    result = _findall(sig, table.mt, table.world, limit)
    result === nothing && return nothing # to many matches
    nr = length(result)
    if nr ≥ 1 && getindex(result, nr).fully_covers
        # no need to fall back to the parent method view
        return MethodMatchResult(result, true)
    end

    parent_result = findall(sig, table.parent; limit)::Union{Nothing, MethodMatchResult}
    parent_result === nothing && return nothing #too many matches

    overlayed = parent_result.overlayed | !isempty(result)
    parent_result = parent_result.matches::MethodLookupResult
    
    # merge the parent match results with the internal method table
    return MethodMatchResult(
    MethodLookupResult(
        vcat(result.matches, parent_result.matches),
        WorldRange(
            max(result.valid_worlds.min_world, parent_result.valid_worlds.min_world),
            min(result.valid_worlds.max_world, parent_result.valid_worlds.max_world)),
        result.ambig | parent_result.ambig),
    overlayed)
end

import Core.Compiler: isoverlayed
isoverlayed(::StackedMethodTable) = true

import Core.Compiler: findsup, _findsup
function findsup(@nospecialize(sig::Type), table::StackedMethodTable)
    match, valid_worlds = _findsup(sig, table.mt, table.world)
    match !== nothing && return match, valid_worlds, true
    # look up in parent
    parent_match, parent_valid_worlds, overlayed = findsup(sig, table.parent)
    return (
        parent_match,
        WorldRange(
            max(valid_worlds.min_world, parent_valid_worlds.min_world),
            min(valid_worlds.max_world, parent_valid_worlds.max_world)),
        overlayed)
end

struct UnionMethodTable{A<:MethodTableView, B<:MethodTableView,} <: MethodTableView
    world::UInt
    mt::MethodTable
    parent_a::A
    parent_b::B
end
UnionMethodTable(world::UInt, mt::MethodTable, parent_a::MethodTable, parent_b::MethodTable) =
    UnionMethodTable(world, mt, StackedMethodTable(world, parent_a), StackedMethodTable(world, parent_b))

function findall(@nospecialize(sig::Type), table::StackedMethodTable; limit::Int=-1)
    result = _findall(sig, table.mt, table.world, limit)
    result === nothing && return nothing # to many matches
    nr = length(result)
    if nr ≥ 1 && getindex(result, nr).fully_covers
        # no need to fall back to the parent method view
        return MethodMatchResult(result, true)
    end

    parent_result_a = findall(sig, table.parent; limit)::Union{Nothing, MethodMatchResult}
    parent_result_a === nothing && return nothing #too many matches

    parent_result_b = findall(sig, table.parent; limit)::Union{Nothing, MethodMatchResult}
    parent_result_b === nothing && return nothing #too many matches

    overlayed = parent_result_a.overlayed || parent_result_b.overlayed || !isempty(result)
    parent_result_a = parent_result_a.matches::MethodLookupResult
    parent_result_b = parent_result_b.matches::MethodLookupResult
    
    # merge the parent match results with the internal method table
    return MethodMatchResult(
    MethodLookupResult(
        unique!(vcat(result.matches, parent_result_a.matches, parent_result_b.matches)),
        WorldRange(
            max(result.valid_worlds.min_world, parent_result_a.valid_worlds.min_world, parent_result_b.valid_worlds.min_world),
            min(result.valid_worlds.max_world, parent_result_a.valid_worlds.max_world, parent_result_b.valid_worlds.min_world)),
        result.ambig || parent_result_a.ambig || parent_result_b.ambig),
    overlayed)
end

import Core.Compiler: isoverlayed
isoverlayed(::UnionMethodTable) = true

import Core.Compiler: findsup, _findsup
function findsup(@nospecialize(sig::Type), table::UnionMethodTable)
    match, valid_worlds = _findsup(sig, table.mt, table.world)
    match !== nothing && return match, valid_worlds, true

    # look up in parents
    parent_match_a, parent_valid_worlds_a, overlayed_a = findsup(sig, table.parent_a)
    parent_match_b, parent_valid_worlds_b, overlayed_b = findsup(sig, table.parent_b)

    if parent_match_a === nothing
        match = parent_match_b
        overlayed = overlayed_b
    elseif parent_match_b === nothing
        match = parent_match_a
        overlayed = overlayed_a
    else
        sig_a = parent_match_a.spec_types
        sig_b = parent_match_b.spec_types
        if sig_a <: sig_b
            match = parent_match_a
            overlayed = overlayed_a
        elseif sig_b <: sig_a
            match = parent_match_b
            overlayed = overlayed_b
        else
            # ambigous? 
            @info "Decide what to do here" parent_match_a parent_match_b
            match = nothing # We found a match in A & B
            overlayed = overlayed_a || overlayed_b
        end
    end
    return (
        match,
        WorldRange(
            max(valid_worlds.min_world, parent_valid_worlds_a.min_world, parent_valid_worlds_b.min_world),
            min(valid_worlds.max_world, parent_valid_worlds_a.max_world, parent_valid_worlds_b.max_world)),
        overlayed)
end


end # module Shenanigans
