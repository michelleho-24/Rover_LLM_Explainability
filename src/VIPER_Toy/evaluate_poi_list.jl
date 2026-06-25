if length(ARGS) < 2
    println("Usage: julia evaluate_poi_list.jl <ice_map.csv> <poi_list>")
    println("  poi_list: 'row,col,dwell;row,col,dwell;...'")
    exit(1)
end

ice_map_file = ARGS[1]
poi_list_str = ARGS[2]

pois = NamedTuple{(:pos, :dwell, :idx), Tuple{Tuple{Int,Int}, Float64, Int}}[]
for (i, triple) in enumerate(split(poi_list_str, ";"))
    parts = split(strip(triple), ",")
    if length(parts) < 2
        println("ERROR: bad POI format '$triple' — expected 'row,col,dwell'")
        exit(1)
    end
    r = parse(Int, strip(parts[1]))
    c = parse(Int, strip(parts[2]))
    d = length(parts) >= 3 ? parse(Float64, strip(parts[3])) : 0.5
    push!(pois, (pos=(r, c), dwell=d, idx=i))
end

using POMDPs, MCTS, Random, POMDPTools, Combinatorics, DelimitedFiles, Distributions, Dates

include(joinpath(@__DIR__, "sim_gp.jl"))
include(joinpath(@__DIR__, "plot.jl"))

var_map_file = replace(ice_map_file, ".csv" => "_variance.csv")
seed = rand(UInt)
Random.seed!(seed)

grid_size = GRID_SIZE
ice_map = load_ice_map_from_csv(ice_map_file)
var_map = load_variance_map_from_csv(var_map_file)
craters = make_craters()
start_pos = START_POS
goal_pos = GOAL_POS

center = (grid_size[1] ÷ 2 + 1, grid_size[2] ÷ 2 + 1)
region_spec = if length(ARGS) >= 3
    shape_arg = ARGS[3]
    if shape_arg == "circle"
        r = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
        circle_region(center, r)
    elseif shape_arg == "rect"
        w = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 50
        h = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 40
        rect_region(center, w, h)
    else
        square_region()
    end
else
    square_region()
end

valid_region = discretize_region(region_spec, grid_size)
region_outside = region_obstacles(valid_region, grid_size)
obstacles = union(craters, region_outside)

init_smg()
sun_ok, sun_msg = validate_sun_coverage()
if !sun_ok
    println("ERROR: $sun_msg")
    exit(1)
end

for p in pois
    if p.pos[1] < 1 || p.pos[1] > grid_size[1] || p.pos[2] < 1 || p.pos[2] > grid_size[2]
        println("ERROR: POI #$(p.idx) at $(p.pos) is out of bounds")
        exit(1)
    end
    if p.pos in obstacles
        println("ERROR: POI #$(p.idx) at $(p.pos) is inside an obstacle / outside region")
        exit(1)
    end
end

n_candidates = N_CANDIDATES
n_drills = N_REQUIRED_DRILLS
beta = UCB_BETA

candidates = choose_candidate_sites(ice_map, var_map, n_candidates, start_pos, obstacles, valid_region;
                                     goal=goal_pos, min_dist=10, beta=beta)

feasible, base_best_time_drills_only, drill_sites, selected_idxs, _ = select_drill_sites(
    candidates, n_drills, start_pos, goal_pos, grid_size, obstacles;
    ice_mean_map=ice_map
)

wg_base = build_waypoint_graph(start_pos, drill_sites, goal_pos, grid_size, obstacles)
drill_nodes_base = collect(2:(n_drills + 1))
goal_node_base = n_drills + 2

base_best_time = Inf
base_best_order = Int[]
for perm in permutations(drill_nodes_base)
    tt = 0.0; cur = 1
    for nxt in perm
        d = waypoint_dist(wg_base, cur, nxt)
        d == typemax(Int) && (tt = Inf; break)
        tt += expected_travel_time(d) + TIME_PER_DRILL; cur = nxt
    end
    tt < Inf && (tt += expected_travel_time(waypoint_dist(wg_base, cur, goal_node_base)))
    if tt < base_best_time
        global base_best_time = tt
        global base_best_order = collect(perm)
    end
end

effective_budget = TIME_BUDGET

println("="^60)
println("BATCH POI EVALUATION — $(length(pois)) POIs")
println("="^60)
println("Drill sites: candidates $selected_idxs at $drill_sites")
println("Base mission time (drills + goal, CBE): $(round(base_best_time, digits=2)) hrs")
println("Time budget: $(TIME_BUDGET) hrs (CBE — run risk_assessment afterward for stochastic margin)")
println()

visit_positions = Tuple{Int,Int}[start_pos]
for n in base_best_order
    push!(visit_positions, wg_base.nodes[n])  # drill positions in order
end
push!(visit_positions, goal_pos)

function segment_time(p1::Tuple{Int,Int}, p2::Tuple{Int,Int}, grid_size, obstacles, elapsed_hours::Float64)
    if p1 == p2
        return 0.0
    end
    path = astar_path_timed(p1, p2, grid_size, obstacles, elapsed_hours)
    if isempty(path) || length(path) < 2
        return Inf
    end
    return path_travel_time_deterministic(path, elapsed_hours).total_hours
end

function compute_total(visit_positions, stops, grid_size, obstacles)
    elapsed = 0.0
    for i in 1:(length(visit_positions) - 1)
        t = segment_time(visit_positions[i], visit_positions[i+1], grid_size, obstacles, elapsed)
        if t == Inf
            return Inf
        end
        elapsed += t + stops[i+1]
    end
    return elapsed
end

stops = Float64[0.0]
for _ in 1:n_drills
    push!(stops, TIME_PER_DRILL)
end
push!(stops, 0.0)

current_total = compute_total(visit_positions, stops, grid_size, obstacles)

accepted = NamedTuple[]
dropped = NamedTuple[]

for poi in pois
    elapsed_at = Float64[0.0]
    for i in 1:(length(visit_positions) - 1)
        t_seg = segment_time(visit_positions[i], visit_positions[i+1], grid_size, obstacles, elapsed_at[end])
        push!(elapsed_at, elapsed_at[end] + t_seg + stops[i+1])
    end

    best_total = Inf
    best_insert = 0
    for insert_after in 1:(length(visit_positions) - 1)
        p_from = visit_positions[insert_after]
        p_to = visit_positions[insert_after + 1]
        elapsed_here = elapsed_at[insert_after]
        t_direct = segment_time(p_from, p_to, grid_size, obstacles, elapsed_here)
        t_to_poi = segment_time(p_from, poi.pos, grid_size, obstacles, elapsed_here)
        if t_to_poi == Inf
            continue
        end
        t_from_poi = segment_time(poi.pos, p_to, grid_size, obstacles, elapsed_here + t_to_poi + poi.dwell)
        if t_from_poi == Inf
            continue
        end
        added = (t_to_poi + poi.dwell + t_from_poi) - t_direct
        new_total = current_total + added
        if new_total < best_total
            best_total = new_total
            best_insert = insert_after
        end
    end

    if best_insert == 0
        push!(dropped, (poi=poi, reason="no reachable insertion point"))
        continue
    end

    if best_total <= effective_budget
        insert!(visit_positions, best_insert + 1, poi.pos)
        insert!(stops, best_insert + 1, poi.dwell)
        global current_total = best_total
        from_label = best_insert == 1 ? "start" : "stop $(best_insert - 1)"
        push!(accepted, (poi=poi, total_after=best_total, insert_after=from_label))
        println("  ✓ POI #$(poi.idx) at $(poi.pos) (dwell $(poi.dwell)h) → inserted after $from_label, total now $(round(best_total, digits=2)) hrs")
    else
        excess = best_total - effective_budget
        push!(dropped, (poi=poi, reason="would exceed budget by $(round(excess, digits=2)) hrs (best total $(round(best_total, digits=2)) hrs)"))
        println("  ✗ POI #$(poi.idx) at $(poi.pos) (dwell $(poi.dwell)h) → DROPPED ($(round(best_total, digits=2)) hrs > $(round(effective_budget, digits=2)) hrs)")
    end
end

println()
println("="^60)
println("SUMMARY: $(length(accepted))/$(length(pois)) POIs fit")
println("="^60)

output_lines = String[]
push!(output_lines, "Batch POI evaluation: $(length(pois)) POIs requested")
push!(output_lines, "Region: $(describe_region(region_spec, length(valid_region), grid_size))")
push!(output_lines, "Drill sites: candidates $selected_idxs at $drill_sites")
push!(output_lines, "Base mission time (drills only, CBE): $(round(base_best_time, digits=2)) hrs")
push!(output_lines, "Effective budget (CBE): $(round(effective_budget, digits=2)) hrs")
push!(output_lines, "Final mission time (CBE, with accepted POIs): $(round(current_total, digits=2)) hrs")
push!(output_lines, "")
push!(output_lines, "ACCEPTED ($(length(accepted))):")
for a in accepted
    push!(output_lines, "  POI #$(a.poi.idx) at $(a.poi.pos) — dwell $(a.poi.dwell)h, inserted after $(a.insert_after)")
end
push!(output_lines, "")
push!(output_lines, "DROPPED ($(length(dropped))):")
for d in dropped
    push!(output_lines, "  POI #$(d.poi.idx) at $(d.poi.pos) — $(d.reason)")
end

if !isempty(dropped)
    push!(output_lines, "")
    push!(output_lines, "OPTIONS to fit more POIs:")
    push!(output_lines, "  1. Reduce dwell times on lower-priority POIs")
    push!(output_lines, "  2. Drop a drill site (replace with a POI via evaluate_poi 'replace' mode — saves ~5 hrs per drill)")
    push!(output_lines, "  3. Try different drill site combinations (use select_custom_sites with other candidates).")
    push!(output_lines, "     Current candidates: $selected_idxs. Other available: $(setdiff(1:n_candidates, selected_idxs))")
    push!(output_lines, "  4. Reorder POIs by priority — current order is the input order.")
end

output_text = join(output_lines, "\n")
println()
println(output_text)

output_path = joinpath(@__DIR__, "..", "..", "data", "poi_list_result.txt")
open(output_path, "w") do f
    write(f, output_text)
end
println("\nResult saved to data/poi_list_result.txt")
