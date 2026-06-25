using DelimitedFiles
using Dates

const SMG_CSV_PATH = joinpath(@__DIR__, "..", "..", "speed_made_good_long_step.csv")
const EPHEMERIS_2025_PATH = joinpath(@__DIR__, "..", "..", "azimuths_elevations_nobile_2025.csv")
const EPHEMERIS_2027_PATH = joinpath(@__DIR__, "..", "..", "azimuths_elevations_nobile_2027.csv")

struct SMGTable
    data::Matrix{Float64}  # 360 × 6, already in m/s
end

function load_smg_table(path::String=SMG_CSV_PATH)::SMGTable
    raw = readdlm(path, ',', Float64; skipstart=1)
    speeds_cm = raw[:, 2:7]
    speeds_m = speeds_cm ./ 100.0
    return SMGTable(speeds_m)
end
function lookup_prospecting_smg(table::SMGTable, sun_theta_deg::Float64)::Float64
    row = 1 + clamp(floor(Int, mod(sun_theta_deg, 360.0)), 0, 359)
    return table.data[row, 3]
end

struct EphemerisEntry
    time::DateTime
    sun_azimuth_deg::Float64
    sun_elevation_deg::Float64
end

struct Ephemeris
    entries::Vector{EphemerisEntry}
end

function load_ephemeris(path::String)::Ephemeris
    entries = EphemerisEntry[]
    open(path) do f
        readline(f)  # skip header
        for line in eachline(f)
            parts = split(line, ',')
            length(parts) < 3 && continue
            t_str = replace(String(parts[1]), "Z" => "")
            t = DateTime(t_str, dateformat"yyyy-mm-ddTHH:MM:SS")
            az = parse(Float64, parts[2])
            el = parse(Float64, parts[3])
            push!(entries, EphemerisEntry(t, az, el))
        end
    end
    return Ephemeris(entries)
end

function load_default_ephemeris()::Ephemeris
    if isfile(EPHEMERIS_2025_PATH)
        return load_ephemeris(EPHEMERIS_2025_PATH)
    elseif isfile(EPHEMERIS_2027_PATH)
        return load_ephemeris(EPHEMERIS_2027_PATH)
    else
        error("No ephemeris CSV found. Expected at $EPHEMERIS_2025_PATH or $EPHEMERIS_2027_PATH")
    end
end
function sun_position_at(eph::Ephemeris, t::DateTime)::Tuple{Float64, Float64}
    entries = eph.entries
    if t <= entries[1].time
        return entries[1].sun_azimuth_deg, entries[1].sun_elevation_deg
    end
    if t >= entries[end].time
        return entries[end].sun_azimuth_deg, entries[end].sun_elevation_deg
    end

    lo, hi = 1, length(entries)
    while hi - lo > 1
        mid = div(lo + hi, 2)
        if entries[mid].time <= t
            lo = mid
        else
            hi = mid
        end
    end

    t0 = entries[lo].time
    t1 = entries[hi].time
    frac = Float64(Dates.value(t - t0)) / Float64(Dates.value(t1 - t0))

    az0, az1 = entries[lo].sun_azimuth_deg, entries[hi].sun_azimuth_deg
    el0, el1 = entries[lo].sun_elevation_deg, entries[hi].sun_elevation_deg

    delta_az = az1 - az0
    if delta_az > 180.0
        delta_az -= 360.0
    elseif delta_az < -180.0
        delta_az += 360.0
    end
    az = mod(az0 + frac * delta_az, 360.0)
    el = el0 + frac * (el1 - el0)

    return az, el
end
function check_sun_coverage(eph::Ephemeris, start_time::DateTime, duration_hours::Float64=24.0)::Tuple{Bool, String}
    end_time = start_time + Millisecond(round(Int, duration_hours * 3600 * 1000))

    if start_time < eph.entries[1].time
        return false, "Start time $(start_time) is before ephemeris data begins ($(eph.entries[1].time))."
    end
    if end_time > eph.entries[end].time
        return false, "Mission end time $(end_time) exceeds ephemeris data (ends $(eph.entries[end].time))."
    end

    check_step = Minute(10)
    t = start_time
    while t <= end_time
        _, el = sun_position_at(eph, t)
        if el <= 0.0
            return false, "Sun is below the horizon at $(t) (elevation = $(round(el, digits=2))°). Choose a start time when the sun remains above the horizon for the full $(duration_hours)h mission."
        end
        t += check_step
    end

    return true, "Sun remains above the horizon for the full mission."
end

function rover_heading_deg(x1::Real, y1::Real, x2::Real, y2::Real)::Float64
    dx = Float64(x2 - x1)
    dy = Float64(y2 - y1)
    if dx == 0.0 && dy == 0.0
        return 0.0
    end
    angle_from_east = atand(dy, dx)  # degrees, CCW from East
    heading = mod(90.0 - angle_from_east, 360.0)
    return heading
end

function compute_step_smg(table::SMGTable, eph::Ephemeris,
                          x1::Real, y1::Real, x2::Real, y2::Real,
                          t::DateTime)::Tuple{Float64, Bool}
    if x1 == x2 && y1 == y2
        return 0.0, false
    end

    sun_az, _ = sun_position_at(eph, t)
    heading = rover_heading_deg(x1, y1, x2, y2)

    sun_theta = mod(heading - sun_az, 360.0)

    smg = lookup_prospecting_smg(table, sun_theta)

    if smg >= 0.001
        return smg, false
    end

    smg_plus  = lookup_prospecting_smg(table, mod(sun_theta + 45.0, 360.0))
    smg_minus = lookup_prospecting_smg(table, mod(sun_theta - 45.0, 360.0))
    tack_smg = cos(deg2rad(45.0)) * (smg_plus + smg_minus) / 2.0

    if tack_smg < 0.001
        tack_smg = 0.001
    end

    return tack_smg, true
end
function step_travel_time(table::SMGTable, eph::Ephemeris,
                          x1::Real, y1::Real, x2::Real, y2::Real,
                          t::DateTime, cell_size_meters::Float64)::Tuple{Float64, Bool}
    smg, is_tacking = compute_step_smg(table, eph, x1, y1, x2, y2, t)

    dx = abs(Float64(x2 - x1))
    dy = abs(Float64(y2 - y1))
    dist_cells = sqrt(dx^2 + dy^2)
    dist_meters = dist_cells * cell_size_meters

    if is_tacking
        dist_meters *= 1.0 / cos(deg2rad(45.0))
    end

    time_seconds = dist_meters / smg
    return time_seconds / 3600.0, is_tacking
end
