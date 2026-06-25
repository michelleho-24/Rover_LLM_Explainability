if length(ARGS) < 4
    println("Usage: julia evaluate_poi.jl <ice_map.csv> <poi_row> <poi_col> <add|replace> [drill_idx] [dwell_hours] [region_shape] [region_params...]")
    exit(1)
end

ice_map_file = ARGS[1]
poi_pos = (parse(Int, ARGS[2]), parse(Int, ARGS[3]))
mode = lowercase(ARGS[4])

replace_drill_idx = 0
dwell_time = 0.5  # default: 30 min observation stop

region_arg_start = 0

if mode == "replace"
    if length(ARGS) < 5
        println("ERROR: replace mode requires drill index (1, 2, or 3)")
        exit(1)
    end
    global replace_drill_idx = parse(Int, ARGS[5])
    if length(ARGS) >= 6
        try
            global dwell_time = parse(Float64, ARGS[6])
            global region_arg_start = 7
        catch
            global region_arg_start = 6
        end
    end
elseif mode == "add"
    if length(ARGS) >= 5
        try
            global dwell_time = parse(Float64, ARGS[5])
            global region_arg_start = 6
        catch
            global region_arg_start = 5
        end
    end
else
    println("ERROR: mode must be 'add' or 'replace', got '$mode'")
    exit(1)
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
region_spec = if region_arg_start > 0 && region_arg_start <= length(ARGS)
    shape_arg = ARGS[region_arg_start]
    if shape_arg == "circle"
        r = region_arg_start + 1 <= length(ARGS) ? parse(Int, ARGS[region_arg_start + 1]) : 30
        circle_region(center, r)
    elseif shape_arg == "rect"
        w = region_arg_start + 1 <= length(ARGS) ? parse(Int, ARGS[region_arg_start + 1]) : 50
        h = region_arg_start + 2 <= length(ARGS) ? parse(Int, ARGS[region_arg_start + 2]) : 40
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

if poi_pos[1] < 1 || poi_pos[1] > grid_size[1] || poi_pos[2] < 1 || poi_pos[2] > grid_size[2]
    println("ERROR: POI position $poi_pos is out of grid bounds $(grid_size)")
    exit(1)
end
if poi_pos in obstacles
    println("ERROR: POI position $poi_pos is inside an obstacle (crater or outside region)")
    exit(1)
end

n_candidates = N_CANDIDATES
n_drills = N_REQUIRED_DRILLS
beta = UCB_BETA

candidates = choose_candidate_sites(ice_map, var_map, n_candidates, start_pos, obstacles, valid_region;
                                     goal=goal_pos, min_dist=10, beta=beta)

feasible, best_time, drill_sites, selected_idxs, _ = select_drill_sites(
    candidates, n_drills, start_pos, goal_pos, grid_size, obstacles;
    ice_mean_map=ice_map
)

println("\n" * "="^60)
println("POI EVALUATION — mode: $(uppercase(mode))")
println("="^60)
println("POI position: $poi_pos")
println("POI dwell time: $dwell_time hrs")
println("Current drill sites: candidates $selected_idxs at $drill_sites")
println("Current best expected time: $(round(best_time, digits=2)) hrs")

output_lines = String[]
n_region = length(valid_region)

if mode == "add"

    all_waypoints = vcat(drill_sites, [poi_pos])
    wg_with_poi = build_waypoint_graph(start_pos, all_waypoints, goal_pos, grid_size, obstacles)
    poi_node = n_drills + 2  # 5
    goal_node_poi = n_drills + 3  # 6

    wg_base = build_waypoint_graph(start_pos, drill_sites, goal_pos, grid_size, obstacles)
    drill_nodes_base = collect(2:(n_drills + 1))
    goal_node_base = n_drills + 2

    base_best_time = Inf
    base_best_order = Int[]
    for perm in permutations(drill_nodes_base)
        tt = 0.0
        cur = 1
        for nxt in perm
            d = waypoint_dist(wg_base, cur, nxt)
            d == typemax(Int) && (tt = Inf; break)
            tt += expected_travel_time(d)
            tt += TIME_PER_DRILL
            cur = nxt
        end
        if tt < Inf
            d = waypoint_dist(wg_base, cur, goal_node_base)
            tt += expected_travel_time(d)
        end
        if tt < base_best_time
            global base_best_time = tt
            global base_best_order = collect(perm)
        end
    end

    visit_sequence = vcat([1], base_best_order, [goal_node_base])

    println("\nBase path (no POI): $(round(base_best_time, digits=2)) hrs")
    println("Exploration budget without POI: $(round(TIME_BUDGET - base_best_time, digits=2)) hrs")

    best_insertion_time = Inf
    best_insertion_pos = 0
    insertion_results = []

    for insert_after in 1:(length(visit_sequence) - 1)
        from_node_base = visit_sequence[insert_after]
        to_node_base = visit_sequence[insert_after + 1]

        from_node = from_node_base
        to_node = to_node_base == goal_node_base ? goal_node_poi : to_node_base

        d_direct = waypoint_dist(wg_with_poi, from_node, to_node)
        t_direct = expected_travel_time(d_direct)

        d_to_poi = waypoint_dist(wg_with_poi, from_node, poi_node)
        d_from_poi = waypoint_dist(wg_with_poi, poi_node, to_node)

        if d_to_poi == typemax(Int) || d_from_poi == typemax(Int)
            push!(insertion_results, (pos=insert_after, time=Inf, added=Inf, feasible=false))
            continue
        end

        t_detour = expected_travel_time(d_to_poi) + dwell_time + expected_travel_time(d_from_poi)
        added_time = t_detour - t_direct
        total_with_poi = base_best_time + added_time

        is_feasible = total_with_poi <= TIME_BUDGET

        from_label = insert_after == 1 ? "start" : "drill$(from_node_base - 1)"
        to_label = to_node_base == goal_node_base ? "goal" : "drill$(to_node_base - 1)"

        push!(insertion_results, (pos=insert_after, time=total_with_poi, added=added_time,
                                   feasible=is_feasible, from=from_label, to=to_label))

        println("  Insert between $from_label → $to_label: +$(round(added_time, digits=2)) hrs → total $(round(total_with_poi, digits=2)) hrs $(is_feasible ? "✓" : "✗")")

        if total_with_poi < best_insertion_time
            global best_insertion_time = total_with_poi
            global best_insertion_pos = insert_after
        end
    end

    push!(output_lines, "Mode: ADD POI to existing route")
    push!(output_lines, "POI position: $poi_pos")
    push!(output_lines, "POI dwell time: $dwell_time hrs")
    push!(output_lines, "Region: $(describe_region(region_spec, n_region, grid_size))")
    push!(output_lines, "Current drill sites: candidates $selected_idxs")
    push!(output_lines, "Base path time (no POI, no exploration): $(round(base_best_time, digits=2)) hrs")
    push!(output_lines, "")

    feasible_insertions = filter(r -> r.feasible, insertion_results)

    if isempty(feasible_insertions)
        push!(output_lines, "Result: INFEASIBLE — no insertion point fits within 24 hours.")
        push!(output_lines, "Best insertion would take $(round(best_insertion_time, digits=2)) hrs ($(round(best_insertion_time - TIME_BUDGET, digits=2)) hrs over budget).")
        push!(output_lines, "")
        push!(output_lines, "Options:")
        push!(output_lines, "  - Reduce POI dwell time (currently $dwell_time hrs)")
        push!(output_lines, "  - Replace a drill site with the POI instead of adding it")
        push!(output_lines, "  - Choose a POI closer to the existing route")
    else
        best = sort(collect(feasible_insertions), by=r -> r.time)[1]

        add_actions = VIPERAction[]
        ordered_targets = Int[]
        for node in base_best_order
            push!(ordered_targets, node - 1)
        end
        poi_target = n_drills + 1
        insert!(ordered_targets, best_insertion_pos, poi_target)

        for t in ordered_targets
            push!(add_actions, drill_action(t))
        end
        push!(add_actions, goto_goal_action())

        all_stops = vcat(drill_sites, [poi_pos])
        add_mdp = VIPER_MDP(wg_with_poi, n_drills + 1, all_stops, 0, obstacles, grid_size, valid_region)

        add_traj = VIPERState[]
        poi_s = VIPERState(1, falses(n_drills + 1), 0, 0, 0.0)
        push!(add_traj, poi_s)
        poi_cur = 1
        for a in add_actions
            dn = a.type == GOTO_GOAL ? (n_drills + 3) : a.target_idx + 1
            d = waypoint_dist(wg_with_poi, poi_cur, dn)
            travel_t = expected_travel_time(d)
            stop_t = a.type == GOTO_GOAL ? 0.0 : (dn == poi_node ? dwell_time : TIME_PER_DRILL)
            new_drilled = copy(poi_s.drilled)
            if a.type == DRILL
                new_drilled[a.target_idx] = true
            end
            global poi_s = VIPERState(dn, new_drilled, poi_s.n_drills_completed + (a.type == DRILL ? 1 : 0),
                                      poi_s.total_steps + d, poi_s.time_elapsed + travel_t + stop_t)
            push!(add_traj, poi_s)
            global poi_cur = dn
        end

        add_enhanced, add_total_steps, add_scanned = plan_exploration(
            wg_with_poi, add_actions, grid_size, obstacles, valid_region
        )

        add_base_segs = Vector{Tuple{Int,Int}}[]
        local add_prev_node = 1
        for a in add_actions
            dn = a.type == GOTO_GOAL ? (n_drills + 3) : a.target_idx + 1
            path = wg_with_poi.paths[add_prev_node, dn]
            push!(add_base_segs, path)
            add_prev_node = dn
        end

        add_full_path = Tuple{Int,Int}[]
        add_path_segs = Vector{Tuple{Int,Int}}[]
        for seg in add_enhanced
            append!(add_full_path, seg)
            push!(add_path_segs, collect(seg))
        end

        local add_travel = 0.0
        local elapsed_add = 0.0
        for (i, seg) in enumerate(add_enhanced)
            seg_start = i == 1 ? start_pos : wg_with_poi.nodes[add_actions[i-1].type == GOTO_GOAL ? (n_drills + 3) : add_actions[i-1].target_idx + 1]
            full_seg = vcat([seg_start], seg)
            stats = path_travel_time_deterministic(full_seg, elapsed_add)
            add_travel += stats.total_hours
            elapsed_add += stats.total_hours
            if i <= length(add_actions) && add_actions[i].type == DRILL
                elapsed_add += (add_actions[i].target_idx == (n_drills + 1) ? dwell_time : TIME_PER_DRILL)
            end
        end
        add_enh_time = add_travel + (n_drills * TIME_PER_DRILL) + dwell_time
        add_coverage = round(100.0 * length(add_scanned) / n_region, digits=1)

        plot_trajectory(add_mdp, add_traj, add_full_path, add_actions, add_path_segs;
            base_segments=add_base_segs, scanned_cells=add_scanned,
            enhanced_steps=add_total_steps, enhanced_time=add_enh_time,
            candidates=candidates, selected_idxs=selected_idxs,
            ice_map=ice_map, var_map=var_map, beta=beta,
            region_spec=region_spec,
            output_filename="trajectory_poi_add.png")

        push!(output_lines, "Result: FEASIBLE")
        push!(output_lines, "Best insertion: between $(best.from) and $(best.to)")
        push!(output_lines, "Added time: +$(round(best.added, digits=2)) hrs")
        push!(output_lines, "Total mission time: $(round(best.time, digits=2)) hrs")
        push!(output_lines, "Enhanced path: $add_total_steps steps")
        push!(output_lines, "Conservative time (with exploration): $(round(add_enh_time, digits=2)) hrs")
        push!(output_lines, "Exploration coverage: $add_coverage%")
        push!(output_lines, "Remaining for exploration: $(round(TIME_BUDGET - best.time, digits=2)) hrs")
        push!(output_lines, "Plot saved: trajectory_poi_add.png")
        push!(output_lines, "")
        if length(feasible_insertions) > 1
            push!(output_lines, "All feasible insertion points:")
            for r in sort(collect(feasible_insertions), by=r -> r.time)
                push!(output_lines, "  Between $(r.from) → $(r.to): $(round(r.time, digits=2)) hrs (+$(round(r.added, digits=2)) hrs)")
            end
        end
    end

elseif mode == "replace"

    if replace_drill_idx < 1 || replace_drill_idx > n_drills
        println("ERROR: drill index must be 1-$n_drills, got $replace_drill_idx")
        exit(1)
    end

    replaced_pos = drill_sites[replace_drill_idx]
    replaced_cand = selected_idxs[replace_drill_idx]

    new_sites = copy(drill_sites)
    new_sites[replace_drill_idx] = poi_pos

    wg_new = build_waypoint_graph(start_pos, new_sites, goal_pos, grid_size, obstacles)
    new_drill_nodes = collect(2:(n_drills + 1))
    new_goal = n_drills + 2

    new_best_time = Inf
    new_best_order = Int[]
    for perm in permutations(new_drill_nodes)
        tt = 0.0
        cur = 1
        for nxt in perm
            d = waypoint_dist(wg_new, cur, nxt)
            d == typemax(Int) && (tt = Inf; break)
            tt += expected_travel_time(d)
            target_idx = nxt - 1
            if target_idx == replace_drill_idx
                tt += dwell_time
            else
                tt += TIME_PER_DRILL
            end
            cur = nxt
        end
        if tt < Inf
            d = waypoint_dist(wg_new, cur, new_goal)
            tt += expected_travel_time(d)
        end
        if tt < new_best_time
            global new_best_time = tt
            global new_best_order = collect(perm)
        end
    end

    new_feasible = new_best_time <= TIME_BUDGET
    time_saved = best_time - new_best_time

    println("\nReplacing drill $replace_drill_idx (candidate $replaced_cand at $replaced_pos) with POI at $poi_pos")
    println("Original time: $(round(best_time, digits=2)) hrs")
    println("New time: $(round(new_best_time, digits=2)) hrs ($(time_saved > 0 ? "saves" : "adds") $(round(abs(time_saved), digits=2)) hrs)")
    println("Feasible: $(new_feasible ? "✓" : "✗")")

    push!(output_lines, "Mode: REPLACE drill site with POI")
    push!(output_lines, "Replaced: drill $replace_drill_idx (candidate $replaced_cand at $replaced_pos)")
    push!(output_lines, "POI position: $poi_pos")
    push!(output_lines, "POI dwell time: $dwell_time hrs (vs $TIME_PER_DRILL hrs for drilling)")
    push!(output_lines, "Region: $(describe_region(region_spec, n_region, grid_size))")
    push!(output_lines, "")

    replaced_ice = ice_map[replaced_pos[1], replaced_pos[2]]
    replaced_unc = sqrt(var_map[replaced_pos[1], replaced_pos[2]])
    push!(output_lines, "Replaced site stats: ice_estimate=$(round(replaced_ice, digits=4)), uncertainty=$(round(replaced_unc, digits=4))")

    if new_feasible
        new_desired_steps = 0
        prev_n = 1
        for node in new_best_order
            global new_desired_steps += waypoint_dist(wg_new, prev_n, node)
            global prev_n = node
        end
        new_desired_steps += waypoint_dist(wg_new, prev_n, new_goal)

        new_mdp = VIPER_MDP(wg_new, n_drills, new_sites, new_desired_steps, obstacles, grid_size, valid_region)
        new_solver = MCTSSolver(n_iterations=MCTS_N_ITERATIONS, depth=MCTS_DEPTH, exploration_constant=MCTS_EXPLORATION_CONSTANT)
        new_policy = solve(new_solver, new_mdp)

        poi_state = rand(initialstate(new_mdp))
        poi_traj = [poi_state]
        poi_actions = VIPERAction[]
        for step in 1:50
            a = action(new_policy, poi_state)
            push!(poi_actions, a)
            global poi_state = rand(transition(new_mdp, poi_state, a))
            push!(poi_traj, poi_state)
            isterminal(new_mdp, poi_state) && break
        end

        poi_enhanced, poi_total_steps, poi_scanned = plan_exploration(
            wg_new, poi_actions, grid_size, obstacles, valid_region
        )

        poi_base_segs = Vector{Tuple{Int,Int}}[]
        local poi_prev_node = 1
        for a in poi_actions
            dn = a.type == GOTO_GOAL ? new_goal : a.target_idx + 1
            path = wg_new.paths[poi_prev_node, dn]
            push!(poi_base_segs, path)
            poi_prev_node = dn
        end

        poi_full_path = Tuple{Int,Int}[]
        poi_path_segs = Vector{Tuple{Int,Int}}[]
        for seg in poi_enhanced
            append!(poi_full_path, seg)
            push!(poi_path_segs, collect(seg))
        end

        poi_n_drills = count(a -> a.type == DRILL, poi_actions)
        local poi_travel = 0.0
        local elapsed_poi = 0.0
        for (i, seg) in enumerate(poi_enhanced)
            seg_start = i == 1 ? start_pos : wg_new.nodes[poi_actions[i-1].type == GOTO_GOAL ? new_goal : poi_actions[i-1].target_idx + 1]
            full_seg = vcat([seg_start], seg)
            stats = path_travel_time_deterministic(full_seg, elapsed_poi)
            poi_travel += stats.total_hours
            elapsed_poi += stats.total_hours
            if i <= length(poi_actions) && poi_actions[i].type == DRILL
                elapsed_poi += TIME_PER_DRILL
            end
        end
        poi_enh_time = poi_travel + poi_n_drills * TIME_PER_DRILL
        poi_coverage = round(100.0 * length(poi_scanned) / n_region, digits=1)

        plot_selected = copy(selected_idxs)

        plot_trajectory(new_mdp, poi_traj, poi_full_path, poi_actions, poi_path_segs;
            base_segments=poi_base_segs, scanned_cells=poi_scanned,
            enhanced_steps=poi_total_steps, enhanced_time=poi_enh_time,
            candidates=candidates, selected_idxs=plot_selected,
            ice_map=ice_map, var_map=var_map, beta=beta,
            region_spec=region_spec,
            output_filename="trajectory_poi_replace.png")

        push!(output_lines, "Result: FEASIBLE")
        push!(output_lines, "New mission time: $(round(new_best_time, digits=2)) hrs")
        push!(output_lines, "Time $(time_saved > 0 ? "saved" : "added"): $(round(abs(time_saved), digits=2)) hrs")
        push!(output_lines, "Enhanced path: $poi_total_steps steps")
        push!(output_lines, "Conservative time (with exploration): $(round(poi_enh_time, digits=2)) hrs")
        push!(output_lines, "Exploration coverage: $poi_coverage%")
        push!(output_lines, "Remaining for exploration: $(round(TIME_BUDGET - new_best_time, digits=2)) hrs")
        push!(output_lines, "")
        push!(output_lines, "Trade-off: you lose the drilling data from candidate $replaced_cand (ice estimate $(round(replaced_ice, digits=4))) but gain the POI observation and $(round(abs(time_saved), digits=2)) hrs of exploration time.")
        push!(output_lines, "Plot saved: trajectory_poi_replace.png")
    else
        push!(output_lines, "Result: INFEASIBLE")
        push!(output_lines, "New mission time: $(round(new_best_time, digits=2)) hrs ($(round(new_best_time - TIME_BUDGET, digits=2)) hrs over budget)")
        push!(output_lines, "")

        push!(output_lines, "Trying other drill sites to replace:")
        for try_idx in 1:n_drills
            try_idx == replace_drill_idx && continue
            try_sites = copy(drill_sites)
            try_sites[try_idx] = poi_pos
            try_wg = build_waypoint_graph(start_pos, try_sites, goal_pos, grid_size, obstacles)
            try_drills = collect(2:(n_drills + 1))
            try_goal = n_drills + 2

            try_best = Inf
            for perm in permutations(try_drills)
                tt = 0.0
                cur = 1
                for nxt in perm
                    d = waypoint_dist(try_wg, cur, nxt)
                    d == typemax(Int) && (tt = Inf; break)
                    tt += expected_travel_time(d)
                    tt += (nxt - 1) == try_idx ? dwell_time : TIME_PER_DRILL
                    cur = nxt
                end
                if tt < Inf
                    d = waypoint_dist(try_wg, cur, try_goal)
                    tt += expected_travel_time(d)
                end
                try_best = min(try_best, tt)
            end

            try_feas = try_best <= TIME_BUDGET
            try_cand = selected_idxs[try_idx]
            try_ice = ice_map[drill_sites[try_idx][1], drill_sites[try_idx][2]]
            push!(output_lines, "  Replace drill $try_idx (candidate $try_cand, ice=$(round(try_ice, digits=4))): $(round(try_best, digits=2)) hrs $(try_feas ? "✓ feasible" : "✗ infeasible")")
        end
    end
end

output_text = join(output_lines, "\n")
println("\n" * output_text)

output_path = joinpath(@__DIR__, "..", "..", "data", "poi_result.txt")
open(output_path, "w") do f
    write(f, output_text)
end
println("\nResult saved to data/poi_result.txt")
