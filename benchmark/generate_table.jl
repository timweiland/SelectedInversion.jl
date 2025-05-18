using Measurements
using DataFrames

mean_selinv_time(name) = mean(SUITE["SelInv_spd_supernodal"][name]).time / 1e9
std_selinv_time(name) = std(SUITE["SelInv_spd_supernodal"][name]).time / 1e9
get_selinv_measurement(name) = measurement(mean_selinv_time(name), std_selinv_time(name))

mean_cho_time(name) = mean(SUITE["Cholesky_supernodal"][name]).time / 1e9
std_cho_time(name) = std(SUITE["Cholesky_supernodal"][name]).time / 1e9
get_cho_measurement(name) = measurement(mean_cho_time(name), std_cho_time(name))

function build_table()
    problems, spd_mat_paths = get_suitesparse_spd()
    Ns = Int64[]
    nnzs = Int64[]
    fac_measurements = Measurement{Float64}[]
    selinv_measurements = Measurement{Float64}[]
    for path in spd_mat_paths
        A = MatrixMarket.mmread(path)
        push!(Ns, size(A, 1))
        push!(nnzs, nnz(A))
    end
    for problem in problems
        push!(fac_measurements, get_cho_measurement(problem))
        push!(selinv_measurements, get_selinv_measurement(problem))
    end

    df = DataFrame(
        "Problem" => problems,
        "N" => Ns,
        "NNZ" => nnzs,
        "Factorization time (sec)" => fac_measurements,
        "SelInv time (sec)" => selinv_measurements,
    )
    sort!(df, [:N])
    return df
end
