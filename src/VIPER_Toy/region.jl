
struct RegionSpec
    shape::Symbol
    center::Tuple{Int,Int}
    radius::Int
    width::Int
    height::Int
    vertices::Vector{Tuple{Int,Int}}
    cell_list::Vector{Tuple{Int,Int}}
end

function RegionSpec(shape::Symbol;
                    center::Tuple{Int,Int}=(0,0),
                    radius::Int=0, width::Int=0, height::Int=0,
                    vertices::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
                    cell_list::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[])
    return RegionSpec(shape, center, radius, width, height, vertices, cell_list)
end

square_region() = RegionSpec(:square)
circle_region(center::Tuple{Int,Int}, radius::Int) =
    RegionSpec(:circle; center=center, radius=radius)
rect_region(center::Tuple{Int,Int}, width::Int, height::Int) =
    RegionSpec(:rectangle; center=center, width=width, height=height)
polygon_region(vertices::Vector{Tuple{Int,Int}}) =
    RegionSpec(:polygon; vertices=vertices)
cells_region(cell_list::Vector{Tuple{Int,Int}}) =
    RegionSpec(:cells; cell_list=cell_list)

function discretize_region(spec::RegionSpec, grid_size::Tuple{Int,Int})
    raw = raw_region_cells(spec, grid_size)
    return Set(filter(c -> 1 <= c[1] <= grid_size[1] && 1 <= c[2] <= grid_size[2], raw))
end

function raw_region_cells(spec::RegionSpec, grid_size::Tuple{Int,Int})
    if spec.shape == :square
        return full_grid(grid_size)
    elseif spec.shape == :circle
        return circle_cells(spec.center, spec.radius)
    elseif spec.shape == :rectangle
        return rect_cells(spec.center, spec.width, spec.height)
    elseif spec.shape == :polygon
        return polygon_cells(spec.vertices)
    elseif spec.shape == :cells
        return spec.cell_list
    else
        error("Unknown region shape: $(spec.shape)")
    end
end

function full_grid(grid_size::Tuple{Int,Int})
    cells = Tuple{Int,Int}[]
    for r in 1:grid_size[1], c in 1:grid_size[2]
        push!(cells, (r, c))
    end
    return cells
end

function circle_cells(center::Tuple{Int,Int}, radius::Int)
    ci, cj = center
    r2 = radius^2
    cells = Tuple{Int,Int}[]
    for di in -radius:radius, dj in -radius:radius
        if di^2 + dj^2 <= r2
            push!(cells, (ci + di, cj + dj))
        end
    end
    return cells
end

function rect_cells(center::Tuple{Int,Int}, w::Int, h::Int)
    ci, cj = center
    half_w = w ÷ 2
    half_h = h ÷ 2
    cells = Tuple{Int,Int}[]
    for di in -half_h:(h - 1 - half_h), dj in -half_w:(w - 1 - half_w)
        push!(cells, (ci + di, cj + dj))
    end
    return cells
end
function polygon_cells(vertices::Vector{Tuple{Int,Int}})
    isempty(vertices) && return Tuple{Int,Int}[]

    rows = [v[1] for v in vertices]
    cols = [v[2] for v in vertices]
    min_r, max_r = minimum(rows), maximum(rows)
    min_c, max_c = minimum(cols), maximum(cols)

    cells = Tuple{Int,Int}[]
    n = length(vertices)
    for r in min_r:max_r, c in min_c:max_c
        if point_in_polygon(r, c, vertices, n)
            push!(cells, (r, c))
        end
    end
    return cells
end
function point_in_polygon(r::Int, c::Int,
                           verts::Vector{Tuple{Int,Int}}, n::Int)
    inside = false
    j = n
    for i in 1:n
        ri, ci = verts[i]
        rj, cj = verts[j]
        if ((ri > r) != (rj > r)) &&
           (c < (cj - ci) * (r - ri) / (rj - ri) + ci)
            inside = !inside
        end
        j = i
    end
    return inside
end

function region_obstacles(valid_cells::Set{Tuple{Int,Int}}, grid_size::Tuple{Int,Int})
    obstacles = Set{Tuple{Int,Int}}()
    for r in 1:grid_size[1], c in 1:grid_size[2]
        if !((r, c) in valid_cells)
            push!(obstacles, (r, c))
        end
    end
    return obstacles
end
function describe_region(spec::RegionSpec, n_cells::Int, grid_size::Tuple{Int,Int})
    total = grid_size[1] * grid_size[2]
    pct = round(100.0 * n_cells / total, digits=1)
    if spec.shape == :square
        return "Full $(grid_size[1])×$(grid_size[2]) square ($n_cells cells, $pct%)"
    elseif spec.shape == :circle
        return "Circle at $(spec.center) r=$(spec.radius) ($n_cells cells, $pct%)"
    elseif spec.shape == :rectangle
        return "Rectangle at $(spec.center) $(spec.width)×$(spec.height) ($n_cells cells, $pct%)"
    elseif spec.shape == :polygon
        return "Polygon with $(length(spec.vertices)) vertices ($n_cells cells, $pct%)"
    elseif spec.shape == :cells
        return "Explicit cell list ($n_cells cells, $pct%)"
    end
end
