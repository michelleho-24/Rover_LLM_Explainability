using POMDPs, MCTS, Random, POMDPTools, Combinatorics, DelimitedFiles, Distributions, Dates

include("sim_gp.jl")

ice_map_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "example_map", "ice_map_62x62.csv")
n_rollouts = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
var_map_file = replace(ice_map_file, ".csv" => "_variance.csv")

Random.seed!(42)

println("Loading SMG table and ephemeris...")
init_smg()

sun_ok, sun_msg = validate_sun_coverage()
if !sun_ok
    open(joinpath(@__DIR__, "..", "..", "data", "monte_carlo_result.txt"), "w") do f
        write(f, "Error: $sun_msg\n")
    end
    println("ERROR: $sun_msg")
    exit(1)
end

grid_size = GRID_SIZE
ice_map = load_ice_map_from_csv(ice_map_file)
var_map = load_variance_map_from_csv(var_map_file)
craters = make_craters()
start_pos = START_POS
goal_pos = GOAL_POS

region_spec = if length(ARGS) >= 3
    shape_arg = ARGS[3]
    center = (grid_size[1] ÷ 2 + 1, grid_size[2] ÷ 2 + 1)
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
drill_node_idxs = collect(2:(n_required_drills + 1))
goal_node = n_required_drills + 2

best_3site_time = Inf
best_3site_order = Int[]
for perm in permutations(drill_node_idxs)
    tt = 0.0; cur = 1
    for nxt in perm
        d = waypoint_dist(wg, cur, nxt)
        d == typemax(Int) && (tt = Inf; break)
        tt += expected_travel_time(d) + TIME_PER_DRILL; cur = nxt
    end
    tt < Inf && (tt += expected_travel_time(waypoint_dist(wg, cur, goal_node)))
    if tt < best_3site_time
        global best_3site_time = tt; global best_3site_order = collect(perm)
    end
end

desired_steps = 0; prev_node = 1
for node in best_3site_order
    global desired_steps += waypoint_dist(wg, prev_node, node); global prev_node = node
end
desired_steps += waypoint_dist(wg, prev_node, goal_node)

mdp = VIPER_MDP(wg, n_required_drills, drill_sites, desired_steps, obstacles, grid_size, valid_region)
solver = MCTSSolver(n_iterations=MCTS_N_ITERATIONS, depth=MCTS_DEPTH, exploration_constant=MCTS_EXPLORATION_CONSTANT)
policy = solve(solver, mdp)

state = rand(initialstate(mdp)); actions_taken = VIPERAction[]
for step in 1:50
    a = action(policy, state); push!(actions_taken, a)
    global state = rand(transition(mdp, state, a))
    isterminal(mdp, state) && break
end

old_mode = SMG_MODE
global SMG_MODE = :deterministic
enhanced_segments, total_enhanced_steps, _ = plan_exploration(
    wg, actions_taken, grid_size, obstacles, valid_region
)
global SMG_MODE = old_mode

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

println("\nRunning $n_rollouts stochastic rollouts...")
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
worst = rollout_times[end]
best = rollout_times[1]

n_over_budget = count(t -> t > TIME_BUDGET, rollout_times)
success_rate = round(100.0 * (n_rollouts - n_over_budget) / n_rollouts, digits=1)

result = """
Monte Carlo Risk Assessment ($n_rollouts rollouts, SMG std_frac=$(SMG_STD_FRAC)):
- Plan: $actions_str
- Enhanced path: $total_enhanced_steps steps
- Deterministic time (CBE): $(round(det_time, digits=2)) hrs
- Stochastic distribution:
    Best:  $(round(best, digits=2)) hrs
    P50:   $(round(p50, digits=2)) hrs
    P75:   $(round(p75, digits=2)) hrs
    P90:   $(round(p90, digits=2)) hrs
    P95:   $(round(p95, digits=2)) hrs
    P99:   $(round(p99, digits=2)) hrs
    Worst: $(round(worst, digits=2)) hrs
- Time budget: $(TIME_BUDGET) hrs
- Success rate (under budget): $success_rate% ($n_over_budget / $n_rollouts exceeded)
- Mission start: $MISSION_START_TIME
- Drill sites: $drill_sites (candidates $selected_idxs)
"""

println(result)

output_path = joinpath(@__DIR__, "..", "..", "data", "monte_carlo_result.txt")
open(output_path, "w") do f
    write(f, result)
end
println("Results saved to data/monte_carlo_result.txt")
