using SuiteSparseMatrixCollection

SPD_mats_names = [
    "bcsstk14",
    "bcsstk24",
    "bcsstk28",
    "bcsstk18",
    "bodyy6",
    "crystm03",
    "wathen120",
    "thermall1",
    "shipsec1",
    "pwtk",
    "parabolic_fem",
    "tmt_sym",
    "ecology2",
    "G3_circuit",
]

SSMC = ssmc_db()

function get_suitesparse_spd()
    SPD_mats = SSMC[
        (SSMC.numerical_symmetry .== 1) .& (SSMC.positive_definite .== true) .& (SSMC.real .== true) .& (SSMC.name .∈ Ref(
            SPD_mats_names,
        )),
        :,
    ]
    paths = fetch_ssmc(SPD_mats, format = "MM")
    return SPD_mats.name,
    [joinpath(path, "$(SPD_mats.name[i]).mtx") for (i, path) in enumerate(paths)]
end
