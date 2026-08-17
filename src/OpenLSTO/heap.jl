# Binary min-heap for the fast marching method (a port of
# `M2DO_LSM/include/heap.h` / `src/heap.cpp`, itself adapted from Scikit-FMM).
#
# `heap` holds list indices; `address[i]` is the grid address of list entry
# `i`; `distance[i]` its priority; `backPointer[i]` maps a list entry back to
# its position in `heap`. `push!` returns the new list index, which the fast
# marching method later passes back to `set_distance!`.

import Base: push!, pop!

mutable struct Heap
    maxLength::Int
    heapLength::Int
    listLength::Int
    distance::Vector{Float64}
    heap::Vector{Int}
    address::Vector{Int}
    backPointer::Vector{Int}
end

function Heap(maxLength::Integer)
    n = Int(maxLength)
    return Heap(n, 0, 0, zeros(n), zeros(Int, n), zeros(Int, n), zeros(Int, n))
end

Base.isempty(heap::Heap) = heap.heapLength == 0
Base.size(heap::Heap) = heap.heapLength

function Base.peek(heap::Heap)
    heap.heapLength != 0 || error("peek: Heap is empty!")
    return heap.distance[heap.heap[1]]
end

function push!(heap::Heap, address::Int, value::Float64)
    heap.heapLength < heap.maxLength || error("push: Heap is full!")
    heap.heapLength += 1
    heap.listLength += 1
    pos = heap.heapLength
    heap.heap[pos] = heap.listLength
    heap.address[heap.listLength] = address
    heap.distance[heap.listLength] = value
    heap.backPointer[heap.listLength] = pos
    sift_down!(heap, 1, pos)
    return heap.listLength
end

function pop!(heap::Heap)
    heap.heapLength != 0 || error("pop: Heap is empty!")
    top = heap.heap[1]
    address = heap.address[top]
    value = heap.distance[top]
    heap.heap[1] = heap.heap[heap.heapLength]
    heap.backPointer[heap.heap[1]] = 1
    heap.heapLength -= 1
    sift_up!(heap, 1)
    return address, value
end

function set_distance!(heap::Heap, index::Int, newDistance::Float64)
    oldDistance = heap.distance[index]
    heap.distance[index] = newDistance
    pos = heap.backPointer[index]
    if newDistance > oldDistance
        sift_up!(heap, pos)
    end
    if heap.distance[heap.heap[pos]] != newDistance
        return nothing
    end
    sift_down!(heap, 1, pos)
    return nothing
end

# Sift the entry at `pos` toward the root.
function sift_down!(heap::Heap, startPos::Int, pos::Int)
    newItem = heap.heap[pos]
    p = pos
    while p > startPos
        parentPos = p >> 1
        parent = heap.heap[parentPos]
        if heap.distance[newItem] < heap.distance[parent]
            heap.heap[p] = parent
            heap.backPointer[parent] = p
            p = parentPos
            continue
        end
        break
    end
    heap.heap[p] = newItem
    return heap.backPointer[newItem] = p
end

# Sift the entry at `pos` toward the leaves.
function sift_up!(heap::Heap, pos::Int)
    startPos = pos
    newItem = heap.heap[pos]
    p = pos
    childPos = 2 * p
    while childPos <= heap.heapLength
        rightPos = childPos + 1
        if rightPos <= heap.heapLength
            if heap.distance[heap.heap[rightPos]] < heap.distance[heap.heap[childPos]]
                childPos = rightPos
            end
        end
        heap.heap[p] = heap.heap[childPos]
        heap.backPointer[heap.heap[childPos]] = p
        p = childPos
        childPos = 2 * p
    end
    heap.heap[p] = newItem
    return sift_down!(heap, startPos, p)
end
