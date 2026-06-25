using POMDPs, MCTS, Random, POMDPTools, Combinatorics, DelimitedFiles, Distributions, Dates

include(joinpath(@__DIR__, "sim_gp.jl"))
include(joinpath(@__DIR__, "plot.jl"))

if length(ARGS) < 4
    println("Usage: julia custom_sites.jl <ice_map.csv> <idx1> <idx2> <idx3> [region_shape] [region_params...]")
    println("  idx1, idx2, idx3 are 1-based indices into the 10 candidate sites.")
    exit(1)
end

ice_map_file = ARGS[1]
var_map_file = replace(ice_map_file, ".csv" => "_variance.csv")
chosen_idxs = [parse(Int, ARGS[i]) for i in 2:4]

seed = rand(UInt)
Random.seed!(seed)

grid_size = GRID_SIZE
ice_map = load_ice_map_from_csv(ice_map_file)
var_map = load_variance_map_from_csv(var_map_file)
craters = make_craters()
start_pos = START_POS
goal_pos = GOAL_POS

center = (grid_size[1] ÷ 2 + 1, grid_size[2] ÷ 2 + 1)
region_spec = if length(ARGS) >= 5
    shape_arg = ARGS[5]
    if shape_arg == "circle"
        r = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 30
        circle_region(center, r)
    elseif shape_arg == "rect"
        w = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 50
        h = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 40
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

println("Region: $(describe_region(region_spec, length(valid_region), grid_size))")

n_candidates = N_CANDIDATES
beta = UCB_BETA

candidates = choose_candidate_sites(ice_map, var_map, n_candidates, start_pos, obstacles, valid_region;
    goal=goal_pos, min_dist=10, beta=beta)

for idx in chosen_idxs
    if idx < 1 || idx > n_candidates
        println("ERROR: Candidate index $idx is out of range (1-$n_candidates)")
        exit(1)
    end
end

println("\n" * "="^60)
println("CUSTOM SITE SELECTION")
println("="^60)
println("Chosen candidates: $chosen_idxs")

for idx in chosen_idxs
    pos = candidates[idx]
    m = ice_map[pos[1], pos[2]]
    s = sqrt(var_map[pos[1], pos[2]])
    println("  Candidate $idx: $pos  ice_estimate=$(round(m, digits=4))  uncertainty=$(round(s, digits=4))")
end

n_drills = N_REQUIRED_DRILLS
chosen_positions = [candidates[i] for i in chosen_idxs]

wg = build_waypoint_graph(start_pos, chosen_positions, goal_pos, grid_size, obstacles)

drill_nodes = collect(2:(n_drills+1))
goal_node = n_drills + 2

best_time = Inf
best_order = Int[]
for perm in permutations(drill_nodes)
    total_time = 0.0
    current = 1
    for next in perm
        d = waypoint_dist(wg, current, next)
        if d == typemax(Int)
            total_time = Inf
            break
        end
        total_time += expected_travel_time(d)
        total_time += TIME_PER_DRILL
        current = next
    end
    if total_time < Inf
        d = waypoint_dist(wg, current, goal_node)
        total_time += expected_travel_time(d)
    end
    if total_time < best_time
        global best_time = total_time
        global best_order = collect(perm)
    end
end

feasible = best_time <= TIME_BUDGET

println("\nBest ordering: $(join([wg.node_labels[i] for i in best_order], " → ")) → goal")
println("Expected time: $(round(best_time, digits=2)) hrs (limit: $(TIME_BUDGET) hrs)")

output_lines = String[]

if feasible
    println("✓ FEASIBLE")

    desired_steps = 0
    prev_node = 1
    for node in best_order
        global desired_steps += waypoint_dist(wg, prev_node, node)
        global prev_node = node
    end
    desired_steps += waypoint_dist(wg, prev_node, goal_node)

    mdp = VIPER_MDP(wg, n_drills, chosen_positions, desired_steps, obstacles, grid_size, valid_region)

    solver = MCTSSolver(n_iterations=MCTS_N_ITERATIONS, depth=MCTS_DEPTH, exploration_constant=MCTS_EXPLORATION_CONSTANT)
    policy = solve(solver, mdp)

    state = rand(initialstate(mdp))
    traj = [state]
    actions_taken = VIPERAction[]

    for step in 1:50
        a = action(policy, state)
        push!(actions_taken, a)
        new_state = rand(transition(mdp, state, a))
        push!(traj, new_state)
        global state = new_state
        if isterminal(mdp, state)
            break
        end
    end

    final = traj[end]
    actions_str = join([a.type == DRILL ? "drill$(a.target_idx)" : "goal" for a in actions_taken], " → ")

    enhanced_segments, total_enhanced_steps, scanned_cells = plan_exploration(
        wg, actions_taken, grid_size, obstacles, valid_region
    )

    base_segments = Vector{Tuple{Int,Int}}[]
    local prev_node_idx = 1
    for a in actions_taken
        dest_node_idx = a.type == GOTO_GOAL ? goal_node : a.target_idx + 1
        path = wg.paths[prev_node_idx, dest_node_idx]
        push!(base_segments, path)
        prev_node_idx = dest_node_idx
    end

    full_path = Tuple{Int,Int}[]
    path_segments = Vector{Tuple{Int,Int}}[]
    for seg in enhanced_segments
        append!(full_path, seg)
        push!(path_segments, collect(seg))
    end

    n_actual_drills = count(a -> a.type == DRILL, actions_taken)
    n_region = length(valid_region)

    local enhanced_travel = 0.0
    local elapsed_enh = 0.0
    for (i, seg) in enumerate(enhanced_segments)
        seg_with_start = i == 1 ? vcat([start_pos], seg) : vcat([wg.nodes[actions_taken[i-1].type == GOTO_GOAL ? goal_node : actions_taken[i-1].target_idx + 1]], seg)
        stats = path_travel_time_deterministic(seg_with_start, elapsed_enh)
        enhanced_travel += stats.total_hours
        elapsed_enh += stats.total_hours
        if i <= length(actions_taken) && actions_taken[i].type == DRILL
            elapsed_enh += TIME_PER_DRILL
        end
    end
    enhanced_time = enhanced_travel + n_actual_drills * TIME_PER_DRILL
    coverage_pct = round(100.0 * length(scanned_cells) / n_region, digits=1)

    plot_trajectory(mdp, traj, full_path, actions_taken, path_segments;
        base_segments=base_segments, scanned_cells=scanned_cells,
        enhanced_steps=total_enhanced_steps, enhanced_time=enhanced_time,
        candidates=candidates, selected_idxs=chosen_idxs,
        ice_map=ice_map, var_map=var_map, beta=beta,
        region_spec=region_spec,
        output_filename="trajectory_custom_sites.png")

    push!(output_lines, "Status: FEASIBLE")
    push!(output_lines, "Region: $(describe_region(region_spec, n_region, grid_size))")
    push!(output_lines, "Chosen candidates: $chosen_idxs")
    push!(output_lines, "Drill positions: $chosen_positions")
    push!(output_lines, "Best visit order: $(join([wg.node_labels[i] for i in best_order], " → ")) → goal")
    push!(output_lines, "Expected time: $(round(best_time, digits=2)) hrs")
    push!(output_lines, "Simulated time: $(round(final.time_elapsed, digits=2)) hrs")
    push!(output_lines, "Enhanced path: $total_enhanced_steps steps")
    push!(output_lines, "Conservative time (with exploration): $(round(enhanced_time, digits=2)) hrs")
    push!(output_lines, "Exploration coverage: $coverage_pct%")
    push!(output_lines, "Actions: $actions_str")
    push!(output_lines, "Mission completed: $(final.current_node == goal_node && final.n_drills_completed >= n_drills && final.time_elapsed <= TIME_BUDGET ? "Yes" : "No")")
    push!(output_lines, "Plot saved: trajectory_custom_sites.png")

else
    println("INFEASIBLE — exceeds 24-hour limit by $(round(best_time - TIME_BUDGET, digits=2)) hrs")

    push!(output_lines, "Status: INFEASIBLE")
    push!(output_lines, "Region: $(describe_region(region_spec, length(valid_region), grid_size))")
    push!(output_lines, "Chosen candidates: $chosen_idxs")
    push!(output_lines, "Drill positions: $chosen_positions")
    push!(output_lines, "Expected time: $(round(best_time, digits=2)) hrs (over budget by $(round(best_time - TIME_BUDGET, digits=2)) hrs)")

    println("\n" * "="^60)
    println("SWAP SUGGESTIONS (keep 2, replace 1)")
    println("="^60)

    remaining = [i for i in 1:n_candidates if !(i in chosen_idxs)]
    swap_suggestions = []

    function find_best_time_for_set(positions)
        wg_test = build_waypoint_graph(start_pos, positions, goal_pos, grid_size, obstacles)
        test_drills = collect(2:(n_drills+1))
        test_goal = n_drills + 2
        bt = Inf
        for perm in permutations(test_drills)
            tt = 0.0
            cur = 1
            for nxt in perm
                d = waypoint_dist(wg_test, cur, nxt)
                if d == typemax(Int)
                    tt = Inf
                    break
                end
                tt += expected_travel_time(d)
                tt += TIME_PER_DRILL
                cur = nxt
            end
            if tt < Inf
                d = waypoint_dist(wg_test, cur, test_goal)
                tt += expected_travel_time(d)
            end
            bt = min(bt, tt)
        end
        return bt
    end

    for drop_pos in 1:3
        kept = [chosen_idxs[j] for j in 1:3 if j != drop_pos]
        dropped = chosen_idxs[drop_pos]

        for swap_in in remaining
            test_idxs = sort(vcat(kept, [swap_in]))
            test_positions = [candidates[i] for i in test_idxs]

            swap_best_time = find_best_time_for_set(test_positions)

            if swap_best_time <= TIME_BUDGET
                m = ice_map[candidates[swap_in][1], candidates[swap_in][2]]
                s = sqrt(var_map[candidates[swap_in][1], candidates[swap_in][2]])
                margin = TIME_BUDGET - swap_best_time

                push!(swap_suggestions, (
                    dropped=dropped,
                    swap_in=swap_in,
                    new_set=test_idxs,
                    time=swap_best_time,
                    margin=margin,
                    ice=m,
                    unc=s
                ))

                println("  Drop candidate $dropped, add candidate $swap_in → $(round(swap_best_time, digits=2)) hrs ($(round(margin, digits=2)) hrs margin)")
                println("    New set: $test_idxs  |  Swap-in ice=$(round(m, digits=4)), uncertainty=$(round(s, digits=4))")
            end
        end
    end

    if isempty(swap_suggestions)
        println("  No feasible single-site swaps found. Consider changing 2 or more sites.")
        push!(output_lines, "Swap suggestions: none (no single-site swap makes this feasible)")
    else
        sort!(swap_suggestions, by=s -> s.margin, rev=true)

        push!(output_lines, "")
        push!(output_lines, "Suggested swaps (keep 2 of your chosen sites, replace 1):")
        for (i, s) in enumerate(swap_suggestions)
            push!(output_lines, "  Option $i: drop candidate $(s.dropped), use candidate $(s.swap_in) instead → sites $(s.new_set)")
            push!(output_lines, "    Expected time: $(round(s.time, digits=2)) hrs ($(round(s.margin, digits=2)) hrs margin)")
            push!(output_lines, "    Swap-in site: ice_estimate=$(round(s.ice, digits=4)), uncertainty=$(round(s.unc, digits=4))")
        end
    end
end

output_text = join(output_lines, "\n")
println("\n" * output_text)

output_path = joinpath(@__DIR__, "..", "..", "data", "custom_sites_result.txt")
open(output_path, "w") do f
    write(f, output_text)
end
println("\nResult saved to data/custom_sites_result.txt")
