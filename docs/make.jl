using SelectedInversion
using Documenter

DocMeta.setdocmeta!(
    SelectedInversion,
    :DocTestSetup,
    :(using SelectedInversion);
    recursive = true,
)

makedocs(;
    modules = [SelectedInversion],
    authors = "Tim Weiland <hello@timwei.land> and contributors",
    sitename = "SelectedInversion.jl",
    format = Documenter.HTML(;
        canonical = "https://timweiland.github.io/SelectedInversion.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/timweiland/SelectedInversion.jl", devbranch = "main")
