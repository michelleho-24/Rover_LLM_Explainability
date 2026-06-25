using POMDPs
using POMDPTools
using MCTS
using Random
using ProgressMeter
using Combinatorics
using DelimitedFiles
using Distributions
using Dates

include("astar.jl")
include("region.jl")
include("smg.jl")
include("mission_constants.jl")
include("hparams.jl")

function stoch_smg(smg_cbe::Float64, rng::AbstractRNG=Random.GLOBAL_RNG)::Float64
    if smg_cbe < 0.0001
        return 0.0
    end
    std = smg_cbe * SMG_STD_FRAC
    sample = smg_cbe + std * randn(rng)
    return clamp(sample, 0.1 * smg_cbe, 1.0 * smg_cbe)
end

const SMG_TABLE = Ref{SMGTable}()
const EPHEMERIS = Ref{Ephemeris}()
const SMG_LOADED = Ref{Bool}(false)

function init_smg()
    if !SMG_LOADED[]
        SMG_TABLE[] = load_smg_table()
        EPHEMERIS[] = load_default_ephemeris()
        SMG_LOADED[] = true
    end
end

function get_smg_table()::SMGTable
    init_smg()
    return SMG_TABLE[]
end

function get_ephemeris()::Ephemeris
    init_smg()
    return EPHEMERIS[]
end
function validate_sun_coverage()::Tuple{Bool, String}
    init_smg()
    return check_sun_coverage(get_ephemeris(), MISSION_START_TIME, TIME_BUDGET)
end
function mission_time_to_datetime(elapsed_hours::Float64)::DateTime
    return MISSION_START_TIME + Millisecond(round(Int, elapsed_hours * 3600 * 1000))
end
function average_prospecting_smg()::Float64
    table = get_smg_table()
    total = 0.0
    for row in 1:360
        total += table.data[row, 3]  # prospecting, low slope
    end
    return total / 360.0
end

struct WaypointGraph
    nodes::Vector{Tuple{Int,Int}}
    dist::Matrix{Int}
    paths::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}
    node_labels::Vector{String}
end

function build_waypoint_graph(start, targets, goal, grid_size, obstacles)
    nodes = vcat([start], targets, [goal])
    n = length(nodes)
    dist = fill(typemax(Int), n, n)
    paths = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for i in 1:n
        dist[i, i] = 0
        for j in (i+1):n
            path = astar_path_timed(nodes[i], nodes[j], grid_size, obstacles, 0.0)
            d = isempty(path) ? typemax(Int) : length(path) - 1
            dist[i, j] = d
            dist[j, i] = d
            if !isempty(path)
                paths[(i, j)] = path
                paths[(j, i)] = reverse(path)
            end
        end
    end

    labels = vcat(["start"], ["drill$i" for i in 1:length(targets)], ["goal"])
    return WaypointGraph(nodes, dist, paths, labels)
end
function waypoint_dist(wg::WaypointGraph, i::Int, j::Int)
    return wg.dist[i, j]
end
function expected_travel_time(dist_steps::Int)
    if dist_steps == typemax(Int)
        return Inf
    end
    avg_speed = average_prospecting_smg()
    dist_meters = dist_steps * CELL_SIZE_METERS
    return dist_meters / avg_speed / 3600.0
end

struct PathStats
    total_hours::Float64
    n_tacking::Int
    n_turns::Int
    min_smg_cms::Float64
    max_smg_cms::Float64
    avg_smg_cms::Float64
end
function path_travel_time(path::Vector{Tuple{Int,Int}}, start_elapsed_hours::Float64)::PathStats
    if length(path) < 2
        return PathStats(0.0, 0, 0, 0.0, 0.0, 0.0)
    end

    table = get_smg_table()
    eph = get_ephemeris()
    total_hours = 0.0
    n_tacking = 0
    n_turns = 0
    smg_values = Float64[]
    prev_heading = -1.0
    current_sampled_smg = -1.0

    for i in 1:(length(path) - 1)
        x1, y1 = path[i]
        x2, y2 = path[i+1]

        heading = rover_heading_deg(x1, y1, x2, y2)
        heading_changed = prev_heading >= 0.0 && abs(mod(heading - prev_heading + 180.0, 360.0) - 180.0) > 0.1
        if heading_changed
            total_hours += TURN_PENALTY_HOURS
            n_turns += 1
        end
        prev_heading = heading

        t = mission_time_to_datetime(start_elapsed_hours + total_hours)
        smg_cbe, is_tacking = compute_step_smg(table, eph, x1, y1, x2, y2, t)

        if SMG_MODE == :stochastic && (heading_changed || current_sampled_smg < 0.0)
            current_sampled_smg = stoch_smg(smg_cbe)
        end

        smg = SMG_MODE == :stochastic ? current_sampled_smg : smg_cbe

        push!(smg_values, smg * 100.0)

        dx = abs(Float64(x2 - x1))
        dy = abs(Float64(y2 - y1))
        dist_cells = sqrt(dx^2 + dy^2)
        dist_meters = dist_cells * CELL_SIZE_METERS
        if is_tacking
            dist_meters *= 1.0 / cos(deg2rad(45.0))
        end
        dt = smg > 0.0 ? (dist_meters / smg / 3600.0) : 0.0
        total_hours += dt

        if is_tacking
            n_tacking += 1
        end
    end

    return PathStats(
        total_hours, n_tacking, n_turns,
        minimum(smg_values), maximum(smg_values),
        sum(smg_values) / length(smg_values)
    )
end
function path_travel_time_deterministic(path::Vector{Tuple{Int,Int}}, start_elapsed_hours::Float64)::PathStats
    if length(path) < 2
        return PathStats(0.0, 0, 0, 0.0, 0.0, 0.0)
    end

    table = get_smg_table()
    eph = get_ephemeris()
    total_hours = 0.0
    n_tacking = 0
    n_turns = 0
    smg_values = Float64[]
    prev_heading = -1.0

    for i in 1:(length(path) - 1)
        x1, y1 = path[i]
        x2, y2 = path[i+1]

        heading = rover_heading_deg(x1, y1, x2, y2)
        if prev_heading >= 0.0 && abs(mod(heading - prev_heading + 180.0, 360.0) - 180.0) > 0.1
            total_hours += TURN_PENALTY_HOURS
            n_turns += 1
        end
        prev_heading = heading

        t = mission_time_to_datetime(start_elapsed_hours + total_hours)
        smg, is_tacking = compute_step_smg(table, eph, x1, y1, x2, y2, t)

        push!(smg_values, smg * 100.0)

        dx = abs(Float64(x2 - x1))
        dy = abs(Float64(y2 - y1))
        dist_cells = sqrt(dx^2 + dy^2)
        dist_meters = dist_cells * CELL_SIZE_METERS
        if is_tacking
            dist_meters *= 1.0 / cos(deg2rad(45.0))
        end
        dt = smg > 0.0 ? (dist_meters / smg / 3600.0) : 0.0
        total_hours += dt

        if is_tacking
            n_tacking += 1
        end
    end

    return PathStats(
        total_hours, n_tacking, n_turns,
        minimum(smg_values), maximum(smg_values),
        sum(smg_values) / length(smg_values)
    )
end

function choose_candidate_sites(
    ice_mean_map::Matrix{Float64},
    ice_var_map::Matrix{Float64},
    n::Int,
    start::Tuple{Int,Int},
    obstacles::Set{Tuple{Int,Int}},
    valid_region::Set{Tuple{Int,Int}};
    goal::Union{Tuple{Int,Int}, Nothing}=nothing,
    min_dist::Int=10,
    beta::Float64=5.0
)
    ucb_map = ice_mean_map .+ beta .* sqrt.(ice_var_map)

    idxs = sortperm(vec(ucb_map), rev=true)
    valid_sites = Tuple{Int,Int}[]

    for i in idxs
        pos = Tuple(CartesianIndices(ucb_map)[i])
        if !(pos in valid_region) || pos in obstacles
            continue
        end
        if goal !== nothing && pos == goal
            continue
        end
        dist_from_start = abs(pos[1] - start[1]) + abs(pos[2] - start[2])
        if dist_from_start < min_dist
            continue
        end
        if goal !== nothing
            dist_from_goal = abs(pos[1] - goal[1]) + abs(pos[2] - goal[2])
            if dist_from_goal < min_dist
                continue
            end
        end
        too_close = false
        for prev_site in valid_sites
            dist_between = abs(pos[1] - prev_site[1]) + abs(pos[2] - prev_site[2])
            if dist_between < min_dist
                too_close = true
                break
            end
        end
        if !too_close
            push!(valid_sites, pos)
        end
        if length(valid_sites) == n
            break
        end
    end
    return valid_sites
end
function select_drill_sites(
    candidates::Vector{Tuple{Int,Int}},
    n_drills::Int,
    start::Tuple{Int,Int},
    goal::Tuple{Int,Int},
    grid_size::Tuple{Int,Int},
    obstacles::Set{Tuple{Int,Int}};
    ice_mean_map::Union{Matrix{Float64}, Nothing}=nothing
)
    n_candidates = length(candidates)

    full_wg = build_waypoint_graph(start, candidates, goal, grid_size, obstacles)

    candidate_idxs = collect(2:(n_candidates + 1))
    goal_idx = n_candidates + 2
    start_idx = 1

    best_time = Inf
    best_ice_score = -Inf
    best_subset = Int[]
    best_order = Int[]

    for subset in combinations(candidate_idxs, n_drills)
        for perm in permutations(subset)
            total_time = 0.0
            current = start_idx

            for next in perm
                d = waypoint_dist(full_wg, current, next)
                if d == typemax(Int)
                    total_time = Inf
                    break
                end
                total_time += expected_travel_time(d)
                total_time += TIME_PER_DRILL
                current = next
            end

            if total_time < Inf
                d = waypoint_dist(full_wg, current, goal_idx)
                total_time += expected_travel_time(d)
            end

            ice_score = 0.0
            if ice_mean_map !== nothing
                for node_idx in subset
                    pos = candidates[node_idx - 1]
                    ice_score += ice_mean_map[pos[1], pos[2]]
                end
            end

            if total_time < best_time || (total_time == best_time && ice_score > best_ice_score)
                best_time = total_time
                best_ice_score = ice_score
                best_subset = collect(subset)
                best_order = collect(perm)
            end
        end
    end

    selected_candidate_idxs = [idx - 1 for idx in best_subset]
    selected_positions = [candidates[i] for i in selected_candidate_idxs]

    feasible = best_time <= TIME_BUDGET
    return feasible, best_time, selected_positions, selected_candidate_idxs, best_order
end

@enum ActionType DRILL GOTO_GOAL

struct VIPERAction
    type::ActionType
    target_idx::Int
end

drill_action(i::Int) = VIPERAction(DRILL, i)
goto_goal_action() = VIPERAction(GOTO_GOAL, 0)

Base.hash(a::VIPERAction, h::UInt) = hash((a.type, a.target_idx), h)
Base.:(==)(a::VIPERAction, b::VIPERAction) = a.type == b.type && a.target_idx == b.target_idx

struct VIPERState
    current_node::Int
    drilled::BitVector
    n_drills_completed::Int
    total_steps::Int
    time_elapsed::Float64
end

function Base.hash(s::VIPERState, h::UInt)
    return hash((s.current_node, s.drilled, s.n_drills_completed,
                 s.total_steps, s.time_elapsed), h)
end

function Base.:(==)(a::VIPERState, b::VIPERState)
    return a.current_node == b.current_node &&
           a.drilled == b.drilled &&
           a.n_drills_completed == b.n_drills_completed &&
           a.total_steps == b.total_steps &&
           a.time_elapsed == b.time_elapsed
end

mutable struct VIPER_MDP <: MDP{VIPERState, VIPERAction}
    wg::WaypointGraph
    n_drills::Int
    drill_positions::Vector{Tuple{Int,Int}}
    desired_steps::Int
    obstacles::Set{Tuple{Int,Int}}
    grid_size::Tuple{Int,Int}
    valid_region::Set{Tuple{Int,Int}}
end

function POMDPs.initialstate(mdp::VIPER_MDP)
    return Deterministic(VIPERState(
        1, falses(mdp.n_drills), 0, 0, 0.0
    ))
end

function POMDPs.actions(mdp::VIPER_MDP, state::VIPERState)
    acts = VIPERAction[]

    for i in 1:mdp.n_drills
        if !state.drilled[i]
            target_node = i + 1
            if target_node != state.current_node
                push!(acts, drill_action(i))
            end
        end
    end

    if state.n_drills_completed >= mdp.n_drills
        goal_node = mdp.n_drills + 2
        if state.current_node != goal_node
            push!(acts, goto_goal_action())
        end
    end

    return acts
end

function POMDPs.transition(mdp::VIPER_MDP, state::VIPERState, action::VIPERAction)
    if action.type == DRILL
        return transition_drill(mdp, state, action.target_idx)
    else
        return transition_goto_goal(mdp, state)
    end
end

function transition_drill(mdp::VIPER_MDP, state::VIPERState, target_idx::Int)
    candidate_node = target_idx + 1
    dist = waypoint_dist(mdp.wg, state.current_node, candidate_node)
    if dist == typemax(Int)
        return Deterministic(state)
    end

    path_key = (state.current_node, candidate_node)
    if haskey(mdp.wg.paths, path_key)
        stats = path_travel_time(mdp.wg.paths[path_key], state.time_elapsed)
        travel_time = stats.total_hours
    else
        avg_speed = average_prospecting_smg()
        travel_time = (dist * CELL_SIZE_METERS) / avg_speed / 3600.0
    end

    new_drilled = copy(state.drilled)
    new_drilled[target_idx] = true

    return Deterministic(VIPERState(
        candidate_node, new_drilled,
        state.n_drills_completed + 1, state.total_steps + dist,
        state.time_elapsed + travel_time + TIME_PER_DRILL
    ))
end

function transition_goto_goal(mdp::VIPER_MDP, state::VIPERState)
    goal_node = mdp.n_drills + 2
    dist = waypoint_dist(mdp.wg, state.current_node, goal_node)
    if dist == typemax(Int)
        return Deterministic(state)
    end

    path_key = (state.current_node, goal_node)
    if haskey(mdp.wg.paths, path_key)
        stats = path_travel_time(mdp.wg.paths[path_key], state.time_elapsed)
        travel_time = stats.total_hours
    else
        avg_speed = average_prospecting_smg()
        travel_time = (dist * CELL_SIZE_METERS) / avg_speed / 3600.0
    end

    return Deterministic(VIPERState(
        goal_node, copy(state.drilled),
        state.n_drills_completed, state.total_steps + dist,
        state.time_elapsed + travel_time
    ))
end

function POMDPs.reward(mdp::VIPER_MDP, state::VIPERState, action::VIPERAction, next_state::VIPERState)
    time_delta = next_state.time_elapsed - state.time_elapsed

    time_cost = -TIME_PENALTY_RATE * time_delta
    fraction_used = clamp(next_state.time_elapsed / TIME_BUDGET, 0.0, 1.5)
    urgency_penalty = -5.0 * fraction_used^2
    shaping = time_cost + urgency_penalty

    if action.type == DRILL
        return 25.0 + shaping
    end

    if action.type == GOTO_GOAL
        goal_node = mdp.n_drills + 2
        if next_state.current_node == goal_node &&
           next_state.n_drills_completed >= mdp.n_drills

            time_remaining = TIME_BUDGET - next_state.time_elapsed
            if time_remaining < 0
                return -100.0 + shaping
            else
                time_bonus = 20.0 * (time_remaining / TIME_BUDGET)
                return 75.0 + time_bonus + shaping
            end
        end
    end

    return shaping
end

function POMDPs.isterminal(mdp::VIPER_MDP, state::VIPERState)
    goal_node = mdp.n_drills + 2
    mission_complete = (state.current_node == goal_node &&
                        state.n_drills_completed >= mdp.n_drills)
    return mission_complete || state.time_elapsed > TIME_BUDGET
end

POMDPs.discount(::VIPER_MDP) = 0.9

function load_ice_map_from_csv(filepath::String)
    println("Loading ice map from: ", filepath)
    ice_map = readdlm(filepath, ',', Float64)
    println("Ice map loaded: size = ", size(ice_map))
    println("Ice map stats: min=$(minimum(ice_map)), max=$(maximum(ice_map)), mean=$(round(sum(ice_map)/length(ice_map), digits=4))")
    return ice_map
end

function load_variance_map_from_csv(filepath::String)
    println("Loading variance map from: ", filepath)
    var_map = readdlm(filepath, ',', Float64)
    println("Variance map loaded: size = ", size(var_map))
    println("Variance stats: min=$(minimum(var_map)), max=$(maximum(var_map))")
    println("Std dev range: $(round(sqrt(minimum(var_map)), digits=4)) to $(round(sqrt(maximum(var_map)), digits=4))")
    return var_map
end

function make_craters()
    craters = Set{Tuple{Int, Int}}()
    for center in [(20, 20), (40, 15)]
        for dr in 0:1, dc in 0:1
            push!(craters, (center[1] + dr, center[2] + dc))
        end
    end
    big_center = (50, 50)
    for dr in 0:4, dc in 0:4
        push!(craters, (big_center[1] + dr, big_center[2] + dc))
    end
    return craters
end

function scan_from(pos::Tuple{Int,Int}, grid_size::Tuple{Int,Int},
                   valid_region::Set{Tuple{Int,Int}})
    cells = Tuple{Int,Int}[]
    pi, pj = pos
    for di in -2:2, dj in -2:2
        if di^2 + dj^2 <= SCAN_RADIUS_SQ
            cell = (pi + di, pj + dj)
            if 1 <= cell[1] <= grid_size[1] && 1 <= cell[2] <= grid_size[2] &&
               cell in valid_region
                push!(cells, cell)
            end
        end
    end
    return cells
end
function scan_from_path(path::Vector{Tuple{Int,Int}}, grid_size::Tuple{Int,Int},
                        valid_region::Set{Tuple{Int,Int}})
    scanned = Set{Tuple{Int,Int}}()
    for pos in path
        for c in scan_from(pos, grid_size, valid_region)
            push!(scanned, c)
        end
    end
    return scanned
end
function conservative_travel_time(dist_steps::Int)
    avg_speed = average_prospecting_smg()
    dist_meters = dist_steps * CELL_SIZE_METERS
    return dist_meters / avg_speed / 3600.0
end
function expected_stochastic_factor()::Float64
    if SMG_STD_FRAC <= 0.0
        return 1.0
    end
    n_sims = 5000
    n_segments = 20  # representative number of heading runs per mission
    time_factors = Float64[]
    for _ in 1:n_sims
        total_speed = 0.0
        for _ in 1:n_segments
            s = clamp(1.0 + SMG_STD_FRAC * randn(), 0.1, 1.0)
            total_speed += 1.0 / s  # time factor for this segment
        end
        push!(time_factors, total_speed / n_segments)
    end
    sort!(time_factors)
    return time_factors[round(Int, 0.90 * n_sims)]  # P90 time multiplier
end
function compute_exploration_budget(base_segments::Vector{Vector{Tuple{Int,Int}}},
                                    n_drills::Int)
    drill_time = n_drills * TIME_PER_DRILL
    stoch_factor = expected_stochastic_factor()

    elapsed = 0.0
    total_travel = 0.0
    for (i, seg) in enumerate(base_segments)
        stats = path_travel_time_deterministic(seg, elapsed)
        seg_time = stats.total_hours
        total_travel += seg_time
        elapsed += seg_time + drill_time / n_drills
    end

    conservative_travel = total_travel * stoch_factor
    min_total_time = drill_time + conservative_travel
    remaining_time = TIME_BUDGET - min_total_time

    return min_total_time, remaining_time
end

function find_loop_detour(
    base_path::Vector{Tuple{Int,Int}},
    i::Int, j::Int,
    grid_size::Tuple{Int,Int},
    obstacles::Set{Tuple{Int,Int}},
    path_set::Set{Tuple{Int,Int}}
)
    departure = base_path[i]
    rejoin = base_path[j]

    forbidden = union(obstacles, setdiff(path_set, Set([departure, rejoin])))

    open_set = Set([departure])
    came_from = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    g_score = Dict{Tuple{Int,Int}, Float64}(departure => 0.0)
    f_score = Dict{Tuple{Int,Int}, Float64}(departure => chebyshev(departure, rejoin))

    max_search = 500
    iterations = 0
    while !isempty(open_set) && iterations < max_search
        iterations += 1
        current = argmin(pos -> get(f_score, pos, Inf), open_set)

        if current == rejoin
            path = [current]
            while haskey(came_from, current)
                current = came_from[current]
                pushfirst!(path, current)
            end
            return path
        end

        delete!(open_set, current)
        ci, cj = current
        for (ni, nj) in [(ci+1,cj), (ci-1,cj), (ci,cj+1), (ci,cj-1),
                         (ci+1,cj+1), (ci+1,cj-1), (ci-1,cj+1), (ci-1,cj-1)]
            neighbor = (ni, nj)
            if !in_bounds(neighbor, grid_size) || neighbor in forbidden
                if neighbor != rejoin
                    continue
                end
            end
            step_cost = (abs(ni - ci) + abs(nj - cj) == 2) ? SQRT2 : 1.0
            tentative_g = g_score[current] + step_cost
            if tentative_g < get(g_score, neighbor, Inf)
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + chebyshev(neighbor, rejoin)
                push!(open_set, neighbor)
            end
        end
    end

    return Tuple{Int,Int}[]
end
function exploration_path(
    base_path::Vector{Tuple{Int,Int}},
    time_budget_hrs::Float64,
    elapsed_hours::Float64,
    grid_size::Tuple{Int,Int},
    obstacles::Set{Tuple{Int,Int}},
    already_scanned::Set{Tuple{Int,Int}},
    valid_region::Set{Tuple{Int,Int}}
)
    if time_budget_hrs < 0.001 || length(base_path) < 3
        scanned = union(already_scanned, scan_from_path(base_path, grid_size, valid_region))
        return base_path, scanned
    end

    scanned = union(already_scanned, scan_from_path(base_path, grid_size, valid_region))
    result_path = copy(base_path)
    path_set = Set(result_path)
    budget_left = time_budget_hrs

    MIN_GAP = 3
    MAX_GAP = 12

    while budget_left > 0.001
        best_score = 0.0
        best_detour = Tuple{Int,Int}[]
        best_i = 0
        best_j = 0
        best_extra_time = 0.0

        n = length(result_path)
        for gap in MIN_GAP:min(MAX_GAP, n - 1)
            for i in 1:(n - gap)
                j = i + gap

                detour = find_loop_detour(result_path, i, j, grid_size, obstacles, path_set)
                if isempty(detour)
                    continue
                end

                replaced_segment = result_path[i:j]
                replaced_time = path_travel_time_deterministic(replaced_segment, elapsed_hours).total_hours
                detour_time = path_travel_time_deterministic(detour, elapsed_hours).total_hours
                extra_time = detour_time - replaced_time

                if extra_time < 0.0001 || extra_time > budget_left
                    continue
                end

                detour_scanned = scan_from_path(detour, grid_size, valid_region)
                new_cells = length(setdiff(detour_scanned, scanned))

                score = new_cells / extra_time
                if score > best_score
                    best_score = score
                    best_detour = detour
                    best_i = i
                    best_j = j
                    best_extra_time = extra_time
                end
            end
        end

        if best_score == 0.0 || isempty(best_detour)
            break
        end

        new_path = vcat(
            result_path[1:best_i],
            best_detour[2:end-1],
            result_path[best_j:end]
        )

        result_path = new_path
        path_set = Set(result_path)
        scanned = union(scanned, scan_from_path(best_detour, grid_size, valid_region))
        budget_left -= best_extra_time
    end

    return result_path, scanned
end

function plan_exploration(
    wg::WaypointGraph,
    actions::Vector{VIPERAction},
    grid_size::Tuple{Int,Int},
    obstacles::Set{Tuple{Int,Int}},
    valid_region::Set{Tuple{Int,Int}}
)
    goal_idx = length(wg.nodes)
    n_drills = count(a -> a.type == DRILL, actions)

    base_segments = Vector{Tuple{Int,Int}}[]
    min_steps_per_seg = Int[]
    seg_times = Float64[]
    segment_labels = String[]

    prev_node = 1
    elapsed = 0.0
    for a in actions
        if a.type == GOTO_GOAL
            dest_node = goal_idx
        else
            dest_node = a.target_idx + 1
        end
        path_key = (prev_node, dest_node)
        segment = haskey(wg.paths, path_key) ? wg.paths[path_key] : [wg.nodes[prev_node], wg.nodes[dest_node]]
        push!(base_segments, segment)
        push!(min_steps_per_seg, length(segment) - 1)

        stats = path_travel_time_deterministic(segment, elapsed)
        push!(seg_times, stats.total_hours)
        elapsed += stats.total_hours
        if a.type == DRILL
            elapsed += TIME_PER_DRILL
        end

        action_label = a.type == DRILL ? "drill$(a.target_idx)" : "goal"
        push!(segment_labels, action_label)
        prev_node = dest_node
    end

    min_total_steps = sum(min_steps_per_seg)
    min_time, remaining_time = compute_exploration_budget(base_segments, n_drills)

    println("\n" * "="^60)
    println("EXPLORATION PLANNING")
    println("="^60)
    println("Minimum path:     $min_total_steps steps")
    println("Min time:         $(round(min_time, digits=2)) hrs")
    println("Time remaining:   $(round(remaining_time, digits=2)) hrs")

    if remaining_time <= 0.001
        println("No budget for exploration — using shortest paths.")
        return base_segments, min_total_steps, Set{Tuple{Int,Int}}()
    end

    println("Exploration budget: $(round(remaining_time, digits=2)) hrs")

    total_seg_time = sum(seg_times)
    time_budgets = [total_seg_time > 0 ? remaining_time * t / total_seg_time : 0.0 for t in seg_times]

    enhanced_segments = Vector{Tuple{Int,Int}}[]
    all_scanned = Set{Tuple{Int,Int}}()
    total_enhanced_steps = 0
    elapsed = 0.0

    for (i, base_seg) in enumerate(base_segments)
        enhanced, all_scanned = exploration_path(
            base_seg, time_budgets[i], elapsed, grid_size, obstacles, all_scanned, valid_region
        )
        push!(enhanced_segments, enhanced[2:end])
        total_enhanced_steps += length(enhanced) - 1

        seg_enhanced_time = path_travel_time_deterministic(enhanced, elapsed).total_hours
        elapsed += seg_enhanced_time
        if i <= length(actions) && actions[i].type == DRILL
            elapsed += TIME_PER_DRILL
        end

        println("  Segment $i ($(segment_labels[i])): $(min_steps_per_seg[i]) → $(length(enhanced)-1) steps | budget=$(round(time_budgets[i]*60, digits=1))min | scanned: $(length(all_scanned)) cells")
    end

    n_region_cells = length(valid_region)
    coverage_pct = round(100.0 * length(all_scanned) / n_region_cells, digits=1)
    println("Total enhanced path: $total_enhanced_steps steps (was $min_total_steps)")
    println("Total cells scanned: $(length(all_scanned)) / $n_region_cells ($coverage_pct%)")

    return enhanced_segments, total_enhanced_steps, all_scanned
end

struct LegStats
    label::String
    steps::Int
    travel_hours::Float64
    min_smg_cms::Float64
    max_smg_cms::Float64
    avg_smg_cms::Float64
    n_tacking::Int
    n_turns::Int
end
function compute_leg_stats(wg::WaypointGraph, actions::Vector{VIPERAction},
                           n_drills::Int)::Vector{LegStats}
    legs = LegStats[]
    elapsed = 0.0
    current_node = 1
    goal_node = n_drills + 2

    for a in actions
        dest_node = a.type == GOTO_GOAL ? goal_node : a.target_idx + 1
        path_key = (current_node, dest_node)
        dist = waypoint_dist(wg, current_node, dest_node)

        from_label = current_node == 1 ? "start" : (current_node == goal_node ? "goal" : "drill$(current_node - 1)")
        to_label = dest_node == goal_node ? "goal" : "drill$(dest_node - 1)"
        label = "$from_label → $to_label"

        if dist == typemax(Int) || !haskey(wg.paths, path_key)
            push!(legs, LegStats(label, dist == typemax(Int) ? 0 : dist, 0.0, 0.0, 0.0, 0.0, 0, 0))
        else
            stats = path_travel_time_deterministic(wg.paths[path_key], elapsed)
            push!(legs, LegStats(label, dist, stats.total_hours,
                                 stats.min_smg_cms, stats.max_smg_cms,
                                 stats.avg_smg_cms, stats.n_tacking, stats.n_turns))
            elapsed += stats.total_hours
            if a.type == DRILL
                elapsed += TIME_PER_DRILL
            end
        end

        current_node = dest_node
    end

    return legs
end
