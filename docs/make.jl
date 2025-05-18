using SelectedInversion
using Documenter

using SparseArrays

DocMeta.setdocmeta!(
    SelectedInversion,
    :DocTestSetup,
    :(using SelectedInversion);
    recursive = true,
)

include("generate_literate.jl")

makedocs(;
    modules = [SelectedInversion],
    authors = "Tim Weiland <hello@timwei.land> and contributors",
    sitename = "SelectedInversion.jl",
    format = Documenter.HTML(;
        canonical = "https://timweiland.github.io/SelectedInversion.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorials/tutorial.md",
        "SupernodalMatrix" => "supernodal_matrix.md",
    ],
)

deploydocs(; repo = "github.com/timweiland/SelectedInversion.jl", devbranch = "main")
