"""
Supernodal elimination tree construction for parallel selected inversion.

The elimination tree captures dependencies between supernodes during the
selected inversion backward sweep. Supernodes at the same level in the tree
can be processed in parallel.
"""

export SupernodalETree, build_supernodal_etree

"""
    SupernodalETree

Supernodal elimination tree with level information for parallel scheduling.

The tree is oriented for the backward sweep in selected inversion:
- Root (last supernode) is at level 0
- Children have higher level numbers than their parents
- Supernodes at the same level have no dependencies on each other

# Fields
- `parent::Vector{Int}`: parent[j] = parent supernode of j (0 if root)
- `children::Vector{Vector{Int}}`: children[j] = child supernodes of j
- `level::Vector{Int}`: level[j] = distance from root (root has level 0)
- `levels::Vector{Vector{Int}}`: levels[k] = supernodes at level k-1 (1-indexed)
- `n_levels::Int`: total number of levels
"""
struct SupernodalETree
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    level::Vector{Int}
    levels::Vector{Vector{Int}}
    n_levels::Int
end

"""
    build_supernodal_etree(Z::SupernodalMatrix) -> SupernodalETree

Construct the supernodal elimination tree from a SupernodalMatrix.

The parent of supernode j is determined by the first row index in `get_Sj(Z, j)`,
which gives the first supernode that depends on j in the backward sweep.
"""
function build_supernodal_etree(Z::SupernodalMatrix)
    n_super = Z.n_super
    parent = zeros(Int, n_super)

    # Build parent relationships
    for j in 1:(n_super - 1)
        Sj = get_Sj(Z, j)
        if !isempty(Sj)
            first_row = Sj[1] + 1  # Convert from 0-indexed
            parent[j] = Z.col_to_super[first_row]
        end
    end
    # Root (last supernode) has no parent

    # Build children lists
    children = [Int[] for _ in 1:n_super]
    for j in 1:(n_super - 1)
        p = parent[j]
        if p > 0
            push!(children[p], j)
        end
    end

    # Compute levels (root at level 0, increasing toward leaves)
    level = _compute_levels(parent, n_super)

    # Group supernodes by level
    levels = _group_by_level(level)

    return SupernodalETree(parent, children, level, levels, length(levels))
end

"""
    _compute_levels(parent::Vector{Int}, n_super::Int) -> Vector{Int}

Compute level of each supernode where root is at level 0.
Process in reverse order to ensure parents are assigned before children.
"""
function _compute_levels(parent::Vector{Int}, n_super::Int)
    level = fill(-1, n_super)
    level[n_super] = 0  # Root at level 0

    # Process in reverse to handle parent-before-child ordering
    for j in (n_super - 1):-1:1
        p = parent[j]
        if p == 0
            # Disconnected node (shouldn't happen for valid factors)
            level[j] = 0
        else
            level[j] = level[p] + 1
        end
    end

    return level
end

"""
    _group_by_level(level::Vector{Int}) -> Vector{Vector{Int}}

Group supernode indices by their level. Returns a vector where
`result[k]` contains all supernodes at level `k-1` (1-indexed).
"""
function _group_by_level(level::Vector{Int})
    max_level = maximum(level)
    levels = [Int[] for _ in 0:max_level]
    for (j, lv) in enumerate(level)
        push!(levels[lv + 1], j)
    end
    return levels
end
