if length(ARGS) < 2
    println("Usage: julia run_scenario.jl <ice_map.csv> <std_frac> [region_shape] [region_params...]")
    println("  std_frac: 0.0 (deterministic) to 0.5 (very high uncertainty)")
    exit(1)
end

ice_map_file = ARGS[1]
std_frac_input = parse(Float64, ARGS[2])

if std_frac_input < 0.0 || std_frac_input > 1.0
    println("ERROR: std_frac must be between 0.0 and 1.0, got $std_frac_input")
    exit(1)
end

ENV["VIPER_SMG_STD_FRAC"] = string(std_frac_input)
ENV["VIPER_SMG_MODE"] = std_frac_input == 0.0 ? "deterministic" : "stochastic"

scenario_name = if std_frac_input == 0.0
    "BEST CASE (deterministic)"
elseif std_frac_input == 0.2
    "NOMINAL (SHERPA default)"
elseif std_frac_input >= 0.4
    "WORST CASE (high uncertainty)"
else
    "CUSTOM ($(Int(round(std_frac_input * 100)))% std)"
end

println("="^60)
println("SCENARIO: $scenario_name — SMG std_frac = $std_frac_input")
println("="^60)

using POMDPs, MCTS, Random, POMDPTools, Combinatorics, DelimitedFiles, Distributions, Dates

include(joinpath(@__DIR__, "sim_gp.jl"))

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

println("Region: $(describe_region(region_spec, length(valid_region), grid_size))")

n_candidates = N_CANDIDATES
n_required_drills = N_REQUIRED_DRILLS
beta = UCB_BETA

candidates = choose_candidate_sites(ice_map, var_map, n_candidates, start_pos, obstacles, valid_region;
                                     goal=goal_pos, min_dist=10, beta=beta)

feasible, best_time, drill_sites, selected_idxs, _ = select_drill_sites(
    candidates, n_required_drills, start_pos, goal_pos, grid_size, obstacles;
    ice_mean_map=ice_map
)

wg = build_waypoint_graph(start_pos, drill_sites, goal_pos, grid_size, obstacles)
drill_nodes = collect(2:(n_required_drills + 1))
goal_node = n_required_drills + 2

best_order_time = Inf
best_order = Int[]
for perm in permutations(drill_nodes)
    tt = 0.0
    cur = 1
    for nxt in perm
        d = waypoint_dist(wg, cur, nxt)
        d == typemax(Int) && (tt = Inf; break)
        tt += expected_travel_time(d)
        tt += TIME_PER_DRILL
        cur = nxt
    end
    if tt < Inf
        d = waypoint_dist(wg, cur, goal_node)
        tt += expected_travel_time(d)
    end
    if tt < best_order_time
        global best_order_time = tt
        global best_order = collect(perm)
    end
end

desired_steps = 0
prev_node = 1
for node in best_order
    global desired_steps += waypoint_dist(wg, prev_node, node)
    global prev_node = node
end
desired_steps += waypoint_dist(wg, prev_node, goal_node)

old_mode = SMG_MODE
global SMG_MODE = :deterministic
mdp = VIPER_MDP(wg, n_required_drills, drill_sites, desired_steps, obstacles, grid_size, valid_region)
solver = MCTSSolver(n_iterations=MCTS_N_ITERATIONS, depth=MCTS_DEPTH, exploration_constant=MCTS_EXPLORATION_CONSTANT)
policy = solve(solver, mdp)

state = rand(initialstate(mdp))
actions_taken = VIPERAction[]
for step in 1:50
    a = action(policy, state)
    push!(actions_taken, a)
    global state = rand(transition(mdp, state, a))
    isterminal(mdp, state) && break
end
global SMG_MODE = old_mode

enhanced_segments, total_enhanced_steps, _ = plan_exploration(
    wg, actions_taken, grid_size, obstacles, valid_region
)

full_segments = Vector{Tuple{Int,Int}}[]
prev_node_idx = 1
for (i, seg) in enumerate(enhanced_segments)
    if i == 1
        push!(full_segments, vcat([start_pos], seg))
    else
        prev_a = actions_taken[i-1]
        prev_dest = prev_a.type == GOTO_GOAL ? goal_node : prev_a.target_idx + 1
        push!(full_segments, vcat([wg.nodes[prev_dest]], seg))
    end
end

n_actual_drills = count(a -> a.type == DRILL, actions_taken)
actions_str = join([a.type == DRILL ? "drill$(a.target_idx)" : "goal" for a in actions_taken], " → ")

det_travel = 0.0; det_elapsed = 0.0
for (i, seg) in enumerate(full_segments)
    stats = path_travel_time_deterministic(seg, det_elapsed)
    global det_travel += stats.total_hours
    global det_elapsed += stats.total_hours
    if i <= length(actions_taken) && actions_taken[i].type == DRILL
        global det_elapsed += TIME_PER_DRILL
    end
end
det_time = det_travel + n_actual_drills * TIME_PER_DRILL

n_rollouts = 100
global SMG_MODE = :stochastic

rollout_times = Float64[]
for r in 1:n_rollouts
    travel = 0.0; elapsed = 0.0
    for (i, seg) in enumerate(full_segments)
        stats = path_travel_time(seg, elapsed)
        travel += stats.total_hours
        elapsed += stats.total_hours
        if i <= length(actions_taken) && actions_taken[i].type == DRILL
            elapsed += TIME_PER_DRILL
        end
    end
    total = travel + n_actual_drills * TIME_PER_DRILL
    push!(rollout_times, total)
end

sort!(rollout_times)
p50 = rollout_times[max(1, round(Int, 0.50 * n_rollouts))]
p75 = rollout_times[max(1, round(Int, 0.75 * n_rollouts))]
p90 = rollout_times[max(1, round(Int, 0.90 * n_rollouts))]
p95 = rollout_times[max(1, round(Int, 0.95 * n_rollouts))]
p99 = rollout_times[max(1, round(Int, 0.99 * n_rollouts))]

n_over_budget = count(t -> t > TIME_BUDGET, rollout_times)
success_rate = round(100.0 * (n_rollouts - n_over_budget) / n_rollouts, digits=1)

println("\n" * "="^60)
println("SCENARIO RESULTS: $scenario_name")
println("="^60)
println("SMG std_frac: $std_frac_input")
println("Plan: $actions_str")
println("Deterministic time (CBE): $(round(det_time, digits=2)) hrs")
println("P50: $(round(p50, digits=2)) hrs | P90: $(round(p90, digits=2)) hrs | P95: $(round(p95, digits=2)) hrs")
println("Success rate: $success_rate%")

output_lines = [
    "Scenario: $scenario_name",
    "SMG std_frac: $std_frac_input (speed sampled from truncated Normal, can only be slower)",
    "Region: $(describe_region(region_spec, length(valid_region), grid_size))",
    "Plan: $actions_str",
    "Enhanced path: $total_enhanced_steps steps",
    "Deterministic time (CBE): $(round(det_time, digits=2)) hrs",
    "Stochastic timing ($n_rollouts rollouts):",
    "  P50: $(round(p50, digits=2)) hrs",
    "  P75: $(round(p75, digits=2)) hrs",
    "  P90: $(round(p90, digits=2)) hrs",
    "  P95: $(round(p95, digits=2)) hrs",
    "  P99: $(round(p99, digits=2)) hrs",
    "Time budget: $(TIME_BUDGET) hrs",
    "Success rate (under budget): $success_rate% ($n_over_budget / $n_rollouts exceeded)",
    "Drill sites: candidates $selected_idxs",
    "Mission start: $MISSION_START_TIME",
]

if success_rate < 100.0
    push!(output_lines, "WARNING: $(round(100.0 - success_rate, digits=1))% of rollouts exceeded the 24-hour limit under this uncertainty level.")
end
if std_frac_input > 0.2
    push!(output_lines, "Note: This scenario has higher uncertainty than nominal (0.2). Plan may need more time margin.")
elseif std_frac_input < 0.2 && std_frac_input > 0.0
    push!(output_lines, "Note: This scenario has lower uncertainty than nominal (0.2). Real conditions may be worse.")
end

output_text = join(output_lines, "\n")
println("\n" * output_text)

output_path = joinpath(@__DIR__, "..", "..", "data", "scenario_result.txt")
open(output_path, "w") do f
    write(f, output_text)
end
println("\nResult saved to data/scenario_result.txt")
