using Luxor

function draw_selinv_logo(;
    filename = "../src/assets/logo.svg",
    size = 170,
    ghost_color = (0.75, 0.75, 0.80, 0.55),
    chrome_color = (0.3, 0.3, 0.35, 0.9),
)
    Drawing(size, size, filename)
    origin()
    background("transparent")

    # Julia colors
    julia_purple = (0.5843, 0.3451, 0.6980)
    julia_red    = (0.7961, 0.2353, 0.2000)
    julia_green  = (0.2196, 0.5961, 0.1490)
    julia_blue   = (0.2510, 0.3882, 0.8471)
    julia_colors = [julia_purple, julia_red, julia_green, julia_blue]

    # Layout: brackets + grid + superscript must fit in `size`
    bracket_width = 6.0
    bracket_pad = 3.0
    super_width = 22.0
    super_height = 16.0

    # Grid parameters
    n = 7
    cell = 15.0
    gap = 2.0
    step = cell + gap
    grid_size = n * step - gap

    # Center everything horizontally
    total_width = bracket_width + bracket_pad + grid_size + bracket_pad + bracket_width + super_width
    left_edge = -total_width / 2

    grid_left = left_edge + bracket_width + bracket_pad
    grid_right = grid_left + grid_size

    # Center grid vertically with room for superscript above
    grid_top = -grid_size / 2 + super_height / 3
    grid_bottom = grid_top + grid_size

    # A realistic symmetric sparsity pattern (L + L')
    selected = Set([
        # Diagonal
        (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7),
        # Sub-diagonal band
        (2,1), (3,2), (4,3), (5,4), (6,5), (7,6),
        # Fill-in entries
        (3,1), (5,2), (5,3), (6,3), (7,4), (7,5),
        # Mirror for symmetry
        (1,2), (2,3), (3,4), (4,5), (5,6), (6,7),
        (1,3), (2,5), (3,5), (3,6), (4,7), (5,7),
    ])

    # Assign colors based on column groups
    function entry_color(r, c)
        group = min(r, c)
        return julia_colors[mod1(group, 4)]
    end

    # Draw the grid cells
    for r in 1:n, c in 1:n
        x = grid_left + (c - 1) * step
        y = grid_top + (r - 1) * step

        if (r, c) in selected
            col = entry_color(r, c)
            setcolor(col[1], col[2], col[3], 1.0)
            rect(Point(x, y), cell, cell, :fill)
        else
            setcolor(ghost_color...)
            rect(Point(x, y), cell, cell, :fill)
        end
    end

    # Draw matrix brackets
    setcolor(chrome_color...)
    setline(2.5)

    bracket_extend = 4.0
    bt = grid_top - bracket_extend
    bb = grid_bottom + bracket_extend
    hook = 5.0

    # Left bracket [
    bl = grid_left - bracket_pad
    line(Point(bl + hook, bt), Point(bl, bt), :stroke)
    line(Point(bl, bt), Point(bl, bb), :stroke)
    line(Point(bl, bb), Point(bl + hook, bb), :stroke)

    # Right bracket ]
    br = grid_right + bracket_pad
    line(Point(br - hook, bt), Point(br, bt), :stroke)
    line(Point(br, bt), Point(br, bb), :stroke)
    line(Point(br, bb), Point(br - hook, bb), :stroke)

    # Draw "-1" superscript outside the bracket, top-right
    fontface("Georgia-Bold")
    fontsize(18)
    setcolor(chrome_color...)
    text("-1", Point(br + 3, bt + 2), halign = :left, valign = :top)

    finish()
    return preview()
end

# Light mode: dark brackets/text on transparent background
draw_selinv_logo(;
    filename = "../src/assets/logo.svg",
    ghost_color = (0.75, 0.75, 0.80, 0.55),
    chrome_color = (0.3, 0.3, 0.35, 0.9),
)

# Dark mode: light brackets/text on transparent background
draw_selinv_logo(;
    filename = "../src/assets/logo-dark.svg",
    ghost_color = (0.75, 0.75, 0.80, 0.55),
    chrome_color = (0.85, 0.85, 0.85, 0.9),
)
