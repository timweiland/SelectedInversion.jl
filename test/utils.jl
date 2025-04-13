using SelectedInversion

export check_supernodal_chunks_equal_dense

function check_supernodal_chunks_equal_dense(
    S::SupernodalMatrix, A::AbstractMatrix
)
    for sup_idx in 1:S.n_super
        chunk = get_chunk(S, sup_idx)
        chunk_row_idcs, chunk_col_idcs = get_row_col_idcs(S, sup_idx)
        if !(A[chunk_row_idcs, chunk_col_idcs] ≈ chunk)
            return false
        end
    end
    return true
end
