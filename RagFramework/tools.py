import os
import sys
import subprocess
import glob as globmod
import re

from RagFramework.prompts.tool_descriptions import TOOL_DESCRIPTIONS


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def read_summary_field(field: str) -> str:
    try:
        path = os.path.join(PROJECT_ROOT, 'data', 'mission_summary.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    if line.lstrip('- ').startswith(field):
                        return line.split(':', 1)[-1].strip()
    except Exception:
        pass
    return ''


def get_ice_map_path() -> str:
    return read_summary_field('Ice map source') or os.path.join('example_map', 'ice_map_62x62.csv')


def get_mission_start_time() -> str:
    return read_summary_field('Mission start time')


def run_julia(script_path: str, args: list, timeout: int = 180,
              propagate_start: bool = True) -> str:
    full_path = os.path.join(PROJECT_ROOT, script_path)
    if not os.path.exists(full_path):
        return f"Error: script not found at {script_path}"
    env = os.environ.copy()
    if propagate_start:
        start = get_mission_start_time()
        if start:
            env['VIPER_MISSION_START'] = start
    try:
        result = subprocess.run(
            ['julia', '--project=.', full_path] + [str(a) for a in args],
            capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_ROOT, env=env,
        )
        if result.returncode != 0:
            err_detail = result.stderr[:800] or result.stdout[-800:]
            return f"Error (exit {result.returncode}): {err_detail}"
        return result.stdout
    except subprocess.TimeoutExpired:
        return f"Error: script timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


COORD_TUPLE_RE = re.compile(r'\((\d+),\s*(\d+)\)')


def swap_tuples_to_xy(text: str) -> str:
    if not text:
        return text
    return COORD_TUPLE_RE.sub(lambda m: f"({m.group(2)}, {m.group(1)})", text)


def read_data_file_raw(filename: str) -> str:
    path = os.path.join(PROJECT_ROOT, 'data', filename)
    if not os.path.exists(path):
        return f"File not found: data/{filename}"
    with open(path, 'r') as f:
        return swap_tuples_to_xy(f.read())


def open_plot(plot_path: str):
    return


def run_simulation(ice_map_path: str = "", region_shape: str = "square",
                   region_params: str = "") -> str:
    if not ice_map_path:
        ice_map_path = get_ice_map_path()

    args = [ice_map_path]
    if region_shape and region_shape != "square":
        args.append(region_shape)
        if region_params:
            args.extend(region_params.split())

    output = run_julia(os.path.join('src', 'VIPER_Toy', 'test_gp.jl'), args)
    if output.startswith("Error"):
        return output

    open_plot('trajectory_plot.png')
    return read_data_file_raw('mission_summary.txt') or output


def run_scenario(scenario: str = "nominal", region_shape: str = "square",
                 region_params: str = "") -> str:
    scenario = (scenario or "nominal").strip().lower()
    std_frac = {"best": 0.0, "nominal": 0.2, "worst": 0.4}.get(scenario)
    if std_frac is None:
        return f"Error: scenario must be 'best', 'nominal', or 'worst' (got {scenario!r})."

    ice_map_path = get_ice_map_path()
    args = [ice_map_path, str(std_frac)]
    if region_shape and region_shape != "square":
        args.append(region_shape)
        if region_params:
            args.extend(region_params.split())

    output = run_julia(os.path.join('src', 'VIPER_Toy', 'run_scenario.jl'), args, timeout=600)
    if output.startswith("Error"):
        return output

    return read_data_file_raw('scenario_result.txt') or output


def set_mission_start(date: str, time: str = "00:00") -> str:
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
        return "Error: date must be YYYY-MM-DD format, e.g. '2025-10-15'."
    if not re.match(r'^\d{2}:\d{2}$', time):
        return "Error: time must be HH:MM format, e.g. '06:00'."

    start_dt = f"{date}T{time}:00"
    env = os.environ.copy()
    env["VIPER_MISSION_START"] = start_dt

    ice_map_path = get_ice_map_path()
    script = os.path.join(PROJECT_ROOT, 'src', 'VIPER_Toy', 'test_gp.jl')
    try:
        result = subprocess.run(
            ['julia', '--project=.', script, ice_map_path],
            capture_output=True, text=True, timeout=180,
            cwd=PROJECT_ROOT, env=env
        )
        if result.returncode != 0:
            err_detail = result.stderr[:800] or result.stdout[-800:]
            return f"Error (exit {result.returncode}): {err_detail}"
    except subprocess.TimeoutExpired:
        return "Error: simulation timed out after 180s"
    except Exception as e:
        return f"Error: {e}"

    open_plot('trajectory_plot.png')
    return read_data_file_raw('mission_summary.txt') or result.stdout


def evaluate_poi(x: int, y: int, mode: str = "add",
                 drill_idx: int = 0, dwell_time: float = 0.5) -> str:
    row, col = y, x
    ice_map_path = get_ice_map_path()
    args = [ice_map_path, str(row), str(col), mode]
    if mode == "replace":
        if drill_idx < 1 or drill_idx > 3:
            return "Error: drill_idx must be 1, 2, or 3 when mode is 'replace'."
        args.append(str(drill_idx))
    args.append(str(dwell_time))

    output = run_julia(os.path.join('src', 'VIPER_Toy', 'evaluate_poi.jl'), args)
    if output.startswith("Error"):
        return output

    result = read_data_file_raw('poi_result.txt')
    if 'Plot saved' in (result or ''):
        plot_name = 'trajectory_poi_add.png' if mode == 'add' else 'trajectory_poi_replace.png'
        open_plot(plot_name)
    return result or output


def evaluate_poi_list(poi_list: str) -> str:
    if not poi_list or not poi_list.strip():
        return "Error: poi_list is required."

    triples = []
    for raw in poi_list.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 2:
            return f"Error: bad POI '{raw}' — expected 'x,y' or 'x,y,dwell'."
        try:
            x = int(parts[0]); y = int(parts[1])
            dwell = float(parts[2]) if len(parts) >= 3 else 0.5
        except ValueError:
            return f"Error: could not parse POI '{raw}'."
        if not (1 <= x <= 62 and 1 <= y <= 62):
            return f"Error: POI ({x},{y}) is out of grid bounds (1-62)."
        triples.append(f"{y},{x},{dwell}")

    julia_arg = ";".join(triples)
    ice_map_path = get_ice_map_path()

    output = run_julia(
        os.path.join('src', 'VIPER_Toy', 'evaluate_poi_list.jl'),
        [ice_map_path, julia_arg],
        timeout=300
    )
    if output.startswith("Error"):
        return output

    return read_data_file_raw('poi_list_result.txt') or output


def replan_with_map(map_path: str) -> str:
    full = os.path.join(PROJECT_ROOT, map_path)
    if not os.path.exists(full):
        return f"Error: map file not found at {map_path}"
    var_path = full.replace('.csv', '_variance.csv')
    if not os.path.exists(var_path):
        return f"Error: variance map not found at {var_path}"

    output = run_julia(os.path.join('src', 'VIPER_Toy', 'test_gp.jl'), [map_path])
    if output.startswith("Error"):
        return output

    open_plot('trajectory_plot.png')
    return read_data_file_raw('mission_summary.txt') or output


def run_arbitrary_region(vertices: str, start: str = "", goal: str = "") -> str:
    try:
        pairs = [v.strip() for v in vertices.split(';') if v.strip()]
        if len(pairs) < 3:
            return "Error: need at least 3 vertices to define a polygon."
        parsed = []
        for p in pairs:
            parts = p.split(',')
            if len(parts) != 2:
                return f"Error: vertex '{p}' is not in x,y format."
            vx, vy = int(parts[0].strip()), int(parts[1].strip())
            if not (1 <= vx <= 62 and 1 <= vy <= 62):
                return f"Error: vertex ({vx},{vy}) is out of grid bounds (1-62)."
            parsed.append((vy, vx))
    except ValueError as e:
        return f"Error parsing vertices: {e}. Use format '1,1;1,62;62,31'."

    if not start or not goal:
        return ("Error: both start and goal positions are required. "
                "Ask the user where the rover should start and finish. "
                "Both must be inside the polygon region.")

    try:
        sx, sy = [int(v.strip()) for v in start.split(',')]
        gx, gy = [int(v.strip()) for v in goal.split(',')]
        sr, sc = sy, sx
        gr, gc = gy, gx
    except ValueError:
        return "Error: start and goal must be in 'x,y' format, e.g. '10,5'."

    ice_map_path = get_ice_map_path()
    vertex_args = [f"{r},{c}" for r, c in parsed]
    args = [ice_map_path, "polygon"] + vertex_args

    env = os.environ.copy()
    env["VIPER_START_POS"] = f"{sr},{sc}"
    env["VIPER_GOAL_POS"] = f"{gr},{gc}"
    start = get_mission_start_time()
    if start:
        env["VIPER_MISSION_START"] = start

    script = os.path.join(PROJECT_ROOT, 'src', 'VIPER_Toy', 'test_gp.jl')
    try:
        result = subprocess.run(
            ['julia', '--project=.', script] + args,
            capture_output=True, text=True, timeout=240,
            cwd=PROJECT_ROOT, env=env,
        )
        if result.returncode != 0:
            err_detail = result.stderr[:800] or result.stdout[-800:]
            return f"Error (exit {result.returncode}): {err_detail}"
    except subprocess.TimeoutExpired:
        return "Error: planner timed out after 240s"
    except Exception as e:
        return f"Error: {e}"

    open_plot('trajectory_plot.png')
    return read_data_file_raw('mission_summary.txt') or result.stdout


def select_custom_sites(site_indices: str) -> str:
    try:
        idxs = [int(x.strip()) for x in site_indices.split(',')]
    except ValueError:
        return "Error: site_indices must be comma-separated integers, e.g. '2,5,8'"
    if len(idxs) != 3:
        return f"Error: need exactly 3 site indices, got {len(idxs)}"
    if not all(1 <= i <= 10 for i in idxs):
        return "Error: all indices must be between 1 and 10"

    ice_map_path = get_ice_map_path()
    args = [ice_map_path] + [str(i) for i in idxs]

    output = run_julia(os.path.join('src', 'VIPER_Toy', 'custom_sites.jl'), args)
    if output.startswith("Error"):
        return output

    result = read_data_file_raw('custom_sites_result.txt')
    if 'Plot saved' in (result or ''):
        open_plot('trajectory_custom_sites.png')
    return result or output


def change_order(site_indices: str) -> str:
    try:
        idxs = [int(x.strip()) for x in site_indices.split(',')]
    except ValueError:
        return "Error: site_indices must be comma-separated integers, e.g. '2,1,3'"
    if len(idxs) != 3:
        return f"Error: need exactly 3 site indices, got {len(idxs)}"
    if not all(1 <= i <= 10 for i in idxs):
        return "Error: all indices must be between 1 and 10"
    if len(set(idxs)) != 3:
        return "Error: indices must be distinct (each candidate visited once)"

    ice_map_path = get_ice_map_path()
    args = [ice_map_path] + [str(i) for i in idxs]

    output = run_julia(os.path.join('src', 'VIPER_Toy', 'change_order.jl'), args)
    if output.startswith("Error"):
        return output

    result = read_data_file_raw('change_order_result.txt')
    return result or output


def risk_assessment(n_rollouts: int = 100, region_shape: str = "square",
                    region_params: str = "") -> str:
    n_rollouts = min(max(n_rollouts, 10), 500)
    ice_map_path = get_ice_map_path()
    args = [ice_map_path, str(n_rollouts)]
    if region_shape and region_shape != "square":
        args.append(region_shape)
        if region_params:
            args.extend(region_params.split())

    output = run_julia(os.path.join('src', 'VIPER_Toy', 'monte_carlo.jl'), args, timeout=300)
    if output.startswith("Error"):
        return output

    return read_data_file_raw('monte_carlo_result.txt') or output


def show_plot(plot_name: str = "trajectory_plot.png") -> str:
    full = os.path.join(PROJECT_ROOT, plot_name)
    if not os.path.exists(full):
        return f"Plot not found: {plot_name}"
    open_plot(plot_name)
    return f"Opened plot: {plot_name}"


def read_source_code(file_path: str) -> str:
    if not (file_path.startswith('src/') or file_path.startswith('RagFramework/')):
        return "Error: can only read files under src/ or RagFramework/"

    full = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.exists(full):
        return f"File not found: {file_path}"

    with open(full, 'r') as f:
        content = f.read()

    if len(content) > 6000:
        content = content[:6000] + f"\n\n[... truncated, {len(content)} total chars]"
    return content


def read_data_file(file_path: str) -> str:
    if '/' in file_path or '\\' in file_path:
        if file_path.startswith('data/'):
            file_path = file_path[5:]
        else:
            return "Error: provide just the filename within data/"

    return read_data_file_raw(file_path)


def search_mission_document(query: str, top_k: int = 3) -> str:
    try:
        from RagFramework.retrieval import Retriever
        retriever = Retriever()
        result = retriever.get_relevant_mission_chunks(query, top_k=top_k)
        if result:
            return result
        return "No relevant sections found in the mission document."
    except Exception as e:
        return f"Error searching mission document: {e}"


def list_available_files(directory: str = "data") -> str:
    allowed = ('data', 'src', 'example_map', 'RagFramework')
    if not any(directory.startswith(a) for a in allowed):
        return f"Error: can only list directories under {allowed}"

    full = os.path.join(PROJECT_ROOT, directory)
    if not os.path.isdir(full):
        return f"Directory not found: {directory}"

    files = sorted(os.listdir(full))
    if not files:
        return f"Directory {directory} is empty."

    lines = [f"Files in {directory}/:"]
    for f in files:
        fp = os.path.join(full, f)
        if os.path.isdir(fp):
            lines.append(f"  {f}/")
        else:
            size = os.path.getsize(fp)
            lines.append(f"  {f}  ({size:,} bytes)")
    return '\n'.join(lines)


for fn in (run_simulation, run_scenario, set_mission_start, evaluate_poi,
           evaluate_poi_list, replan_with_map, run_arbitrary_region,
           select_custom_sites, change_order, risk_assessment,
           show_plot, read_source_code, read_data_file,
           search_mission_document, list_available_files):
    fn.__doc__ = TOOL_DESCRIPTIONS.get(fn.__name__, "")
