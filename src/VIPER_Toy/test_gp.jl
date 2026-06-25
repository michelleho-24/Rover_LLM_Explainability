using POMDPs, MCTS, Random, POMDPTools, Combinatorics, DelimitedFiles, Distributions, Dates

include("sim_gp.jl")
include("plot.jl")

ice_map_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "example_map", "ice_map_62x62.csv")
var_map_file = replace(ice_map_file, ".csv" => "_variance.csv")

if !@isdefined(seed) || seed === nothing
    seed = rand(UInt)
end
println("RNG seed used: ", seed)
Random.seed!(seed)

println("\nMission start time: $MISSION_START_TIME")
println("SMG mode: $SMG_MODE (std_frac=$SMG_STD_FRAC)")
println("Loading SMG table and ephemeris...")
init_smg()

sun_ok, sun_msg = validate_sun_coverage()
if !sun_ok
    println("\nERROR: $sun_msg")
    println("Simulation aborted — sun must be above the horizon for the full $(TIME_BUDGET)h mission.")
    open(joinpath(@__DIR__, "..", "..", "data", "mission_summary.txt"), "w") do f
        write(f, "Simulation aborted: $sun_msg\n")
    end
    exit(1)
end
println("Sun check: $sun_msg")
println("Average prospecting SMG: $(round(average_prospecting_smg() * 100, digits=2)) cm/s")

grid_size = GRID_SIZE
ice_map = load_ice_map_from_csv(ice_map_file)
var_map = load_variance_map_from_csv(var_map_file)
craters = make_craters()
function parse_pos_env(name, default)
    env = get(ENV, name, "")
    isempty(env) && return default
    parts = split(env, ",")
    return (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])))
end

start_pos = parse_pos_env("VIPER_START_POS", START_POS)
goal_pos = parse_pos_env("VIPER_GOAL_POS", GOAL_POS)

center = (grid_size[1] ÷ 2 + 1, grid_size[2] ÷ 2 + 1)  # (32, 32)

region_spec = if length(ARGS) >= 2
    shape_arg = ARGS[2]
    if shape_arg == "square"
        square_region()
    elseif shape_arg == "circle"
        r = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
        circle_region(center, r)
    elseif shape_arg == "rect"
        w = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
        h = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 40
        rect_region(center, w, h)
    elseif shape_arg == "polygon"
        verts = Tuple{Int,Int}[]
        for i in 3:length(ARGS)
            parts = split(ARGS[i], ",")
            push!(verts, (parse(Int, parts[1]), parse(Int, parts[2])))
        end
        polygon_region(verts)
    else
        println("Unknown shape '$shape_arg', using full square.")
        square_region()
    end
else
    square_region()
end

valid_region = discretize_region(region_spec, grid_size)
region_outside = region_obstacles(valid_region, grid_size)
obstacles = union(craters, region_outside)

println("\n" * "="^60)
println("SCIENCE REGION")
println("="^60)
println("  $(describe_region(region_spec, length(valid_region), grid_size))")
println("  Valid cells: $(length(valid_region))")
println("  Obstacles (craters + outside): $(length(obstacles))")

if !(start_pos in valid_region)
    error("Start position $start_pos is outside the science region!")
end
if !(goal_pos in valid_region)
    error("Goal position $goal_pos is outside the science region!")
end

n_candidates = N_CANDIDATES
n_required_drills = N_REQUIRED_DRILLS
beta = UCB_BETA

candidates = choose_candidate_sites(ice_map, var_map, n_candidates, start_pos, obstacles, valid_region;
                                     goal=goal_pos, min_dist=10, beta=beta)

println("\n" * "="^60)
println("CANDIDATE SITES (UCB, β=$beta)")
println("="^60)
println("  #   Position     Mean     Std Dev    UCB Score")
println("  " * "-"^52)
for (i, pos) in enumerate(candidates)
    m = ice_map[pos[1], pos[2]]
    v = var_map[pos[1], pos[2]]
    s = sqrt(v)
    ucb = m + beta * s
    println("  $(lpad(i, 2))  $(rpad(pos, 10))  $(lpad(round(m, digits=4), 7))   $(lpad(round(s, digits=4), 7))   $(lpad(round(ucb, digits=4), 7))")
end

println("\n" * "="^60)
println("SITE SELECTION (C($(n_candidates),$(n_required_drills)) × $(n_required_drills)! = $(binomial(n_candidates, n_required_drills) * factorial(n_required_drills)) orderings)")
println("="^60)

feasible, best_time, drill_sites, selected_idxs, best_order_full = select_drill_sites(
    candidates, n_required_drills, start_pos, goal_pos, grid_size, obstacles;
    ice_mean_map=ice_map
)

println("Selected drill sites (from candidates):")
for (i, idx) in enumerate(selected_idxs)
    pos = candidates[idx]
    m = ice_map[pos[1], pos[2]]
    s = sqrt(var_map[pos[1], pos[2]])
    ucb = m + beta * s
    println("  Drill $i: candidate #$idx at $pos  mean=$(round(m, digits=4))  std=$(round(s, digits=4))  UCB=$(round(ucb, digits=4))")
end

if feasible
    println("✓ FEASIBLE — best expected time: $(round(best_time, digits=2)) hrs")
else
    println("✗ INFEASIBLE — minimum expected time: $(round(best_time, digits=2)) hrs (limit: $(TIME_BUDGET))")
    println("  Proceeding with MCTS anyway...")
end

println("\nBuilding waypoint graph ($(n_required_drills + 2) nodes)...")
wg = build_waypoint_graph(start_pos, drill_sites, goal_pos, grid_size, obstacles)

println("\nPairwise A* distances:")
n_nodes = length(wg.nodes)
for i in 1:n_nodes
    for j in (i+1):n_nodes
        d = wg.dist[i, j]
        d_str = d == typemax(Int) ? "∞" : "$d steps"
        println("  $(wg.node_labels[i]) → $(wg.node_labels[j]): $d_str")
    end
end

drill_node_idxs = collect(2:(n_required_drills + 1))
goal_node = n_required_drills + 2

best_3site_time = Inf
best_3site_order = Int[]
for perm in permutations(drill_node_idxs)
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
    if total_time < best_3site_time
        global best_3site_time = total_time
        global best_3site_order = collect(perm)
    end
end

desired_steps = 0
prev_node = 1
for node in best_3site_order
    global desired_steps += waypoint_dist(wg, prev_node, node)
    global prev_node = node
end
desired_steps += waypoint_dist(wg, prev_node, goal_node)

order_labels = [wg.node_labels[i] for i in best_3site_order]
println("\nBest ordering: $(join(order_labels, " → ")) → goal")
println("Desired trajectory length: $desired_steps steps")

mdp = VIPER_MDP(
    wg, n_required_drills, drill_sites,
    desired_steps, obstacles, grid_size, valid_region
)

if get(ENV, "VIPER_SWEEP_SETUP_ONLY", "") == "1"
    error("VIPER_SWEEP_HALT")
end

println("\nSolving with MCTS...")
global SMG_MODE = :stochastic
solver = MCTSSolver(
    n_iterations=MCTS_N_ITERATIONS,
    depth=MCTS_DEPTH,
    exploration_constant=MCTS_EXPLORATION_CONSTANT,
)
policy = solve(solver, mdp)

global SMG_MODE = :deterministic
println("\n" * "="^60)
println("DECISION TIMELINE")
println("="^60)

state = rand(initialstate(mdp))
traj = [state]
actions_taken = VIPERAction[]

for step in 1:50
    a = action(policy, state)
    push!(actions_taken, a)
    new_state = rand(transition(mdp, state, a))
    push!(traj, new_state)

    if a.type == DRILL
        pos = drill_sites[a.target_idx]
        println("  Step $(lpad(step, 2)): DRILL  site $(lpad(a.target_idx, 2)) at $pos  [$(round(new_state.time_elapsed, digits=2)) hrs, +5.5h drill]")
    else
        println("  Step $(lpad(step, 2)): GOTO_GOAL  [$(round(new_state.time_elapsed, digits=2)) hrs]")
    end

    global state = new_state
    if isterminal(mdp, state)
        println("  Terminal state reached.")
        break
    end
end

final = traj[end]

println("\n" * "="^60)
println("MISSION RESULTS")
println("="^60)
println("Total time:       $(round(final.time_elapsed, digits=2)) hrs (limit: $(TIME_BUDGET))")
println("Total distance:   $(final.total_steps) steps")
println("Drills completed: $(final.n_drills_completed) / $n_required_drills")
println("Reached goal:     $(final.current_node == goal_node)")
println("Mission OK:       $(final.current_node == goal_node && final.n_drills_completed >= n_required_drills && final.time_elapsed <= TIME_BUDGET)")

actions_str = join([a.type == DRILL ? "drill$(a.target_idx)" : "goal"
                    for a in actions_taken], " → ")
println("Actions:          $actions_str")

global SMG_MODE = :deterministic
println("\nPlanning exploration detours...")
enhanced_segments, total_enhanced_steps, scanned_cells = plan_exploration(
    wg, actions_taken, grid_size, obstacles, valid_region
)

base_segments = Vector{Tuple{Int,Int}}[]
prev_node_idx = 1
for a in actions_taken
    if a.type == GOTO_GOAL
        dest_node = goal_node
    else
        dest_node = a.target_idx + 1
    end
    path_key = (prev_node_idx, dest_node)
    path = haskey(wg.paths, path_key) ? wg.paths[path_key] : [wg.nodes[prev_node_idx], wg.nodes[dest_node]]
    push!(base_segments, path)
    global prev_node_idx = dest_node
end

full_path = Tuple{Int,Int}[]
path_segments = Vector{Tuple{Int,Int}}[]
for seg in enhanced_segments
    append!(full_path, seg)
    push!(path_segments, collect(seg))
end

n_actual_drills = count(a -> a.type == DRILL, actions_taken)

enhanced_travel = 0.0
elapsed_for_enh = 0.0
for (i, seg) in enumerate(enhanced_segments)
    stats = path_travel_time(seg, elapsed_for_enh)
    global enhanced_travel += stats.total_hours
    global elapsed_for_enh += stats.total_hours
    if i <= length(actions_taken) && actions_taken[i].type == DRILL
        global elapsed_for_enh += TIME_PER_DRILL
    end
end
enhanced_time = enhanced_travel + n_actual_drills * TIME_PER_DRILL

println("\nEnhanced path time (actual SMG): $(round(enhanced_time, digits=2)) hrs")
println("Full enhanced path: $(length(full_path)) cells")

plot_trajectory(mdp, traj, full_path, actions_taken, path_segments;
    base_segments=base_segments, scanned_cells=scanned_cells,
    enhanced_steps=total_enhanced_steps, enhanced_time=enhanced_time,
    candidates=candidates, selected_idxs=selected_idxs,
    ice_map=ice_map, var_map=var_map, beta=beta,
    region_spec=region_spec,
    output_filename=get(ENV, "VIPER_TRAJ_OUT", "trajectory_plot.png"))

n_region = length(valid_region)
coverage_pct = round(100.0 * length(scanned_cells) / n_region, digits=1)

avg_smg_cms = round(average_prospecting_smg() * 100, digits=2)

leg_stats = compute_leg_stats(wg, actions_taken, n_required_drills)

leg_lines = String[]
all_leg_smgs = Float64[]
total_tacking = 0
total_turns = 0
for ls in leg_stats
    push!(leg_lines, "  $(ls.label): $(ls.steps) steps, $(round(ls.travel_hours, digits=2)) hrs | SMG avg=$(round(ls.avg_smg_cms, digits=2)) min=$(round(ls.min_smg_cms, digits=2)) max=$(round(ls.max_smg_cms, digits=2)) cm/s | tacking=$(ls.n_tacking) turns=$(ls.n_turns)")
    if ls.avg_smg_cms > 0
        push!(all_leg_smgs, ls.avg_smg_cms)
    end
    global total_tacking += ls.n_tacking
    global total_turns += ls.n_turns
end
mission_avg_smg = isempty(all_leg_smgs) ? 0.0 : round(sum(all_leg_smgs) / length(all_leg_smgs), digits=2)
leg_summary = join(leg_lines, "\n")

mission_summary = """
Mission Summary (8-Dir, Long Step SMG, Turn Penalty):
- Region: $(describe_region(region_spec, n_region, grid_size))
- Ice map source: $ice_map_file
- Mission start time: $MISSION_START_TIME
- SMG mode: $SMG_MODE (std_frac=$SMG_STD_FRAC)
- Average prospecting SMG (table): $(avg_smg_cms) cm/s
- Actual mission avg SMG: $(mission_avg_smg) cm/s
- Total tacking steps: $total_tacking
- Total turns (107s each): $total_turns
- RNG Seed: $seed
- Candidates generated: $n_candidates (min_dist=10)
- Selected drill sites: candidates $selected_idxs
- Optimal set expected time: $(round(best_time, digits=2)) hrs
- Required drills: $n_required_drills
- Total mission time: $(round(final.time_elapsed, digits=2)) hours (limit: $(TIME_BUDGET) hours)
- Total distance traveled: $(final.total_steps) steps (MCTS shortest)
- Enhanced path distance: $total_enhanced_steps steps (with exploration)
- Conservative time estimate: $(round(enhanced_time, digits=2)) hours
- Cells scanned: $(length(scanned_cells)) / $n_region ($coverage_pct%)
- Drills completed: $(final.n_drills_completed) / $n_required_drills
- Mission completed: $(final.current_node == goal_node && final.n_drills_completed >= n_required_drills && final.time_elapsed <= TIME_BUDGET ? "Yes" : "No")
- Time remaining (MCTS): $(round(TIME_BUDGET - final.time_elapsed, digits=2)) hours
- Actions taken: $actions_str

Per-leg speed breakdown:
$leg_summary

Drill site positions: $drill_sites
"""

println("\n" * "="^60)
println("SPEED MADE GOOD")
println("="^60)
sun_start_az, sun_start_el = sun_position_at(get_ephemeris(), MISSION_START_TIME)
sun_end_az, sun_end_el = sun_position_at(get_ephemeris(), MISSION_START_TIME + Millisecond(round(Int, final.time_elapsed * 3600 * 1000)))
println("  Sun azimuth: $(round(sun_start_az, digits=1))° → $(round(sun_end_az, digits=1))° over mission")
println("  Table avg: $avg_smg_cms cm/s | Mission avg: $mission_avg_smg cm/s | Tacking: $total_tacking | Turns: $total_turns ($(round(total_turns * TURN_PENALTY_HOURS * 60, digits=1)) min)")
for ls in leg_stats
    println("  $(ls.label): avg=$(round(ls.avg_smg_cms, digits=2)) min=$(round(ls.min_smg_cms, digits=2)) max=$(round(ls.max_smg_cms, digits=2)) cm/s  tacking=$(ls.n_tacking) turns=$(ls.n_turns)")
end

summary_path = joinpath(@__DIR__, "..", "..", "data", "mission_summary.txt")
open(summary_path, "w") do f
    write(f, mission_summary)
end
println("\nMission summary saved to data/mission_summary.txt")
