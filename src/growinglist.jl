using Base.Threads: Atomic, SpinLock, atomic_add!

# ListSegment represents a segment from a GrowingList; it has a list `data` to hold the elements, filled up to `currentlength`, and possibly a reference to the next segment, if it is not the final segment.
mutable struct ListSegment{T}
    data::Vector{T}
    currentlength::Int
    next::Base.RefValue{ListSegment{T}}
end
ListSegment{T}(data::Vector{T}, currentlength::Int) where T =
    ListSegment{T}(data, currentlength, Ref{ListSegment{T}}())

# getindex, assumes that index is inbounds, traverses the linked list
function _unsafe_getindex(l::ListSegment, i::Int)
    if i <= l.currentlength
        getindex(l.data, i)
    else
        _unsafe_getindex(l.next[], i - l.currentlength)
    end
end

# get or push a new element at the end; in itself not thread safe, should be protected by the lock in the parent GrowingList
function _unsafe_get!(l::ListSegment{T}, n::Int, default, newlength) where T
    N = length(l.data)
    if n > N
        if isassigned(l.next)
            return _unsafe_get!(l.next[], n - N, default, newlength)
        else
            newsegment = Vector{T}(undef, newlength)
            newsegment[1] = default
            l.next = Ref(ListSegment{T}(newsegment, 1))
            return default
        end
    else
        if n <= l.currentlength
            @inbounds return getindex(l.data, n)
        else
            @assert n == l.currentlength+1
            l.data[n] = default
            l.currentlength += 1
            return default
        end
    end
end

"""
    GrowingList{T} <: AbstractVector{T}

    GrowingList([iter,]; sizehint = max(16, length(iter)), growthfactor = 2.)

A thread safe vector / list data structure where new elements can be added at the back.
Once an element is set, it cannot be changed or removed. This ensures thread safe
`getindex` of that element without requiring a lock. The `length` of a `GrowingList`
instance can also be probed without a lock, but the return value will be a lower bound,
i.e. the list can already have increased in length at the same time.

New elements can be added using the syntax
`get!(l::GrowingList, i::Int, value)`
`get!(value_generator::Callable, l::GrowingList, i::Int)`

where the new element `value` or `value_generator()` will only be added if `i` is
`length(l)+1`. If multiple tasks or threads try to `get!` the same index `i`, only one of
them will actually be adding that element. The `value` or `value_generator()` produced by
the different threads should be the same to avoid unpredictable results.

The list is grown by adding new segments using a linked list data structure. This guarantees that existing data does never have to move in memory, which is required in order to make `getindex` threadsafe without lock.
"""
mutable struct GrowingList{T} <: AbstractVector{T}
    first::ListSegment{T}
    totallength::Atomic{Int}
    growthfactor::Float64
    lock::SpinLock
    function GrowingList{T}(iter;
                            sizehint = max(16, length(iter)),
                            growthfactor = 2.) where {T}
        firstsegment = Vector{T}(undef, sizehint)
        i = 0
        next = iterate(iter)
        @inbounds while i < sizehint && next !== nothing
            i += 1
            val, state = next
            firstsegment[i] = val
            next = iterate(iter, state)
        end
        first = ListSegment{T}(firstsegment, i)
        while next !== nothing
            i += 1
            val, state = next
            _unsafe_get!(first, i, val, ceil(Int, (growthfactor - 1) * (i - 1)))
            next = iterate(iter, state)
        end
        return new{T}(first, Atomic{Int}(i), growthfactor, SpinLock())
    end
end
GrowingList(v::Vector{T}; sizehint = max(16, length(v)), growthfactor = 2.) where {T} =
    GrowingList{T}(v; sizehint = sizehint, growthfactor = growthfactor)

GrowingList{T}(; sizehint = 16, growthfactor = 2.) where {T} =
    GrowingList{T}((); sizehint = sizehint, growthfactor = growthfactor)

GrowingList(; sizehint = 16, growthfactor = 2.) =
    GrowingList{Any}((); sizehint = sizehint, growthfactor = growthfactor)

Base.length(l::GrowingList) = l.totallength[]
function _raise_length!(l::GrowingList)
    atomic_add!(l.totallength, 1)
end

Base.size(l::GrowingList) = (length(l),)

@inline function Base.getindex(l::GrowingList, n::Int)
    @boundscheck checkbounds(l, n)
    return _unsafe_getindex(l.first, n)
end

function Base.get!(l::GrowingList, n::Int, default)
    if n <= length(l)
        return _unsafe_getindex(l.first, n)
    else
        ll = l.lock
        lock(ll)
        len = length(l)
        nextlen = len + 1
        if n <= len # try again, maybe already ok now
            unlock(ll)
            return _unsafe_getindex(l.first, n)
        elseif n == nextlen
            _unsafe_get!(l.first, n, default, ceil(Int, (l.growthfactor-1)*len))
            _raise_length!(l)
            unlock(ll)
            return default
        else
            unlock(ll)
            _inserterror(nextlen)
        end
    end
end

@noinline _inserterror(len::Int) =
    throw(ArgumentError("can only insert new element at next index: " * string(len)))

function Base.get!(default::Base.Callable, l::GrowingList, n::Int)
    if n <= l.totallength[]
        return _unsafe_getindex(l.first, n)
    else
        v = default()
        ll = l.lock
        lock(ll)
        len = length(l)
        nextlen = len + 1
        if n <= len # try again, maybe already ok now
            unlock(ll)
            return _unsafe_getindex(l.first, n)
        elseif n == nextlen
            _unsafe_get!(l.first, n, v, ceil(Int, (l.growthfactor-1)*len))
            _raise_length!(l)
            unlock(ll)
            return v
        else
            unlock(ll)
            _inserterror(nextlen)
        end
    end
end
