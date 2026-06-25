using Plots

const SQRT2 = sqrt(2.0)

function astar_path(start::Tuple{Int,Int}, goal::Tuple{Int,Int}, grid_size::Tuple{Int,Int}, craters::Set{Tuple{Int,Int}})
    open_set = Set([start])
    came_from = Dict{Tuple{Int,Int},Tuple{Int,Int}}()

    g_score = Dict{Tuple{Int,Int},Float64}(start => 0.0)
    f_score = Dict{Tuple{Int,Int},Float64}(start => chebyshev(start, goal))

    function neighbors(pos)
        i, j = pos
        candidates = [
            (i+1, j), (i-1, j), (i, j+1), (i, j-1),       # cardinal
            (i+1, j+1), (i+1, j-1), (i-1, j+1), (i-1, j-1) # diagonal
        ]
        return filter(p -> in_bounds(p, grid_size) && !(p in craters), candidates)
    end

    function step_cost(from::Tuple{Int,Int}, to::Tuple{Int,Int})
        di = abs(to[1] - from[1])
        dj = abs(to[2] - from[2])
        return (di + dj == 2) ? SQRT2 : 1.0
    end

    while !isempty(open_set)
        current = argmin(pos -> get(f_score, pos, Inf), open_set)
        if current == goal
            return reconstruct_path(came_from, current)
        end

        delete!(open_set, current)
        for neighbor in neighbors(current)
            tentative_g = g_score[current] + step_cost(current, neighbor)
            if tentative_g < get(g_score, neighbor, Inf)
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + chebyshev(neighbor, goal)
                push!(open_set, neighbor)
            end
        end
    end

    return []  # no path found
end

function reconstruct_path(came_from, current)
    path = [current]
    while haskey(came_from, current)
        current = came_from[current]
        pushfirst!(path, current)
    end
    return path
end

function chebyshev(a::Tuple{Int,Int}, b::Tuple{Int,Int})
    dx = abs(a[1] - b[1])
    dy = abs(a[2] - b[2])
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)
end

function manhattan(a::Tuple{Int,Int}, b::Tuple{Int,Int})
    return abs(a[1] - b[1]) + abs(a[2] - b[2])
end

function in_bounds(pos::Tuple{Int,Int}, grid_size::Tuple{Int,Int})
    i, j = pos
    return 1 ≤ i ≤ grid_size[1] && 1 ≤ j ≤ grid_size[2]
end
function astar_path_timed(start::Tuple{Int,Int}, goal::Tuple{Int,Int},
    grid_size::Tuple{Int,Int}, craters::Set{Tuple{Int,Int}},
    start_elapsed_hours::Float64)
    table = get_smg_table()
    eph = get_ephemeris()
    turn_penalty = TURN_PENALTY_HOURS
    cell_size = CELL_SIZE_METERS

    sun_az, _ = sun_position_at(eph, mission_time_to_datetime(start_elapsed_hours))
    max_smg = 0.0
    for h in 0.0:45.0:315.0
        sun_theta = mod(h - sun_az, 360.0)
        spd = lookup_prospecting_smg(table, sun_theta)
        max_smg = max(max_smg, spd)
    end
    if max_smg < 0.0001
        max_smg = 0.0001
    end

    start_state = (start, -1.0)  # -1.0 = no previous heading

    open_set = Set([start_state])
    came_from = Dict{Tuple{Tuple{Int,Int},Float64},Tuple{Tuple{Int,Int},Float64}}()
    g_score = Dict{Tuple{Tuple{Int,Int},Float64},Float64}(start_state => 0.0)
    f_score = Dict{Tuple{Tuple{Int,Int},Float64},Float64}()

    heuristic_time = chebyshev(start, goal) * cell_size / max_smg / 3600.0
    f_score[start_state] = heuristic_time

    function get_neighbors(pos)
        i, j = pos
        candidates = [
            (i+1, j), (i-1, j), (i, j+1), (i, j-1),
            (i+1, j+1), (i+1, j-1), (i-1, j+1), (i-1, j-1)
        ]
        return filter(p -> in_bounds(p, grid_size) && !(p in craters), candidates)
    end

    while !isempty(open_set)
        current_state = argmin(s -> get(f_score, s, Inf), open_set)
        current_pos, current_heading = current_state

        if current_pos == goal
            path = [current_pos]
            st = current_state
            while haskey(came_from, st)
                st = came_from[st]
                pushfirst!(path, st[1])
            end
            return path
        end

        delete!(open_set, current_state)
        current_g = g_score[current_state]
        current_time = start_elapsed_hours + current_g

        for neighbor in get_neighbors(current_pos)
            x1, y1 = current_pos
            x2, y2 = neighbor

            heading = rover_heading_deg(x1, y1, x2, y2)

            step_time = 0.0
            if current_heading >= 0.0 && abs(mod(heading - current_heading + 180.0, 360.0) - 180.0) > 0.1
                step_time += turn_penalty
            end

            t = mission_time_to_datetime(current_time + step_time)
            smg, is_tacking = compute_step_smg(table, eph, x1, y1, x2, y2, t)

            dx = abs(Float64(x2 - x1))
            dy = abs(Float64(y2 - y1))
            dist_cells = sqrt(dx^2 + dy^2)
            dist_meters = dist_cells * cell_size
            if is_tacking
                dist_meters *= 1.0 / cos(deg2rad(45.0))
            end
            step_time += smg > 0.0 ? (dist_meters / smg / 3600.0) : 1000.0

            neighbor_state = (neighbor, heading)
            tentative_g = current_g + step_time

            if tentative_g < get(g_score, neighbor_state, Inf)
                came_from[neighbor_state] = current_state
                g_score[neighbor_state] = tentative_g
                h = chebyshev(neighbor, goal) * cell_size / max_smg / 3600.0
                f_score[neighbor_state] = tentative_g + h
                push!(open_set, neighbor_state)
            end
        end
    end

    return Tuple{Int,Int}[]
end
