using Plots

const VIPER_TRAJ_TIKZ_FLAG = get(ENV, "VIPER_TRAJ_TIKZ", "0") == "1"
if VIPER_TRAJ_TIKZ_FLAG
    using PGFPlotsX
    pgfplotsx()
end

function plot_trajectory(mdp::VIPER_MDP, traj::Vector{VIPERState},
                         full_path::Vector{Tuple{Int,Int}},
                         actions_taken::Vector{VIPERAction},
                         segments::Vector{Vector{Tuple{Int,Int}}};
                         base_segments::Vector{Vector{Tuple{Int,Int}}}=Vector{Tuple{Int,Int}}[],
                         scanned_cells::Set{Tuple{Int,Int}}=Set{Tuple{Int,Int}}(),
                         enhanced_steps::Int=0,
                         enhanced_time::Float64=0.0,
                         candidates::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
                         selected_idxs::Vector{Int}=Int[],
                         pois::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
                         ice_map::Matrix{Float64}=Matrix{Float64}(undef, 0, 0),
                         var_map::Matrix{Float64}=Matrix{Float64}(undef, 0, 0),
                         beta::Float64=5.0,
                         region_spec::Union{RegionSpec, Nothing}=nothing,
                         output_filename::String="trajectory_plot.png")
    size_x, size_y = mdp.grid_size
    wg = mdp.wg

    start_pos = wg.nodes[1]
    goal_pos  = wg.nodes[end]

    final = traj[end]

    final_steps = enhanced_steps > 0 ? enhanced_steps : final.total_steps
    base_steps = final.total_steps
    display_time = enhanced_time > 0.0 ? enhanced_time : final.time_elapsed
    n_drilled = final.n_drills_completed
    n_region = length(mdp.valid_region)
    coverage_pct = round(100.0 * length(scanned_cells) / n_region, digits=1)

    shape_label = region_spec !== nothing ? string(region_spec.shape) : "square"
    title_str = "$shape_label | $(n_drilled) drills | Time: $(round(display_time, digits=2))h | Base: $(base_steps) → Enh: $(final_steps) | Scan: $coverage_pct% of $n_region cells"

    no_title = get(ENV, "VIPER_TRAJ_NOTITLE", "0") == "1"
    dpi_setting = parse(Int, get(ENV, "VIPER_TRAJ_DPI", "100"))
    legend_mode = get(ENV, "VIPER_TRAJ_LEGEND", "auto")
    legend_only = get(ENV, "VIPER_TRAJ_LEGEND_ONLY", "0") == "1"
    show_legend = legend_only || (no_title && legend_mode == "outside_left")
    legend_pos = legend_only ? :inside : (show_legend ? :outerleft : (no_title ? false : :topleft))
    p = if VIPER_TRAJ_TIKZ_FLAG
        plot(
            aspect_ratio=1,
            xlim=(0.5, size_y + 0.5),
            ylim=(0.5, size_x + 0.5),
            legend=false,
            grid=false,
            framestyle=:none,
            xlabel="", ylabel="", title="",
            ticks=nothing,
            showaxis=false,
        )
    else
        plot(
            xlabel=legend_only ? "" : (no_title ? "X (m)" : "X"),
            ylabel=legend_only ? "" : (no_title ? "Y (m)" : "Y"),
            title=no_title ? "" : title_str,
            aspect_ratio=legend_only ? :none : 1,
            xlim=legend_only ? (1e6, 1e6+1) : (0.5, size_y + 0.5),
            ylim=legend_only ? (1e6, 1e6+1) : (0.5, size_x + 0.5),
            legend=legend_pos,
            legendfontsize=show_legend ? 18 : 5,
            titlefontsize=8,
            tickfontsize=legend_only ? 1 : (no_title ? 20 : 8),
            guidefontsize=legend_only ? 1 : (no_title ? 22 : 10),
            dpi=dpi_setting,
            grid=legend_only ? false : (no_title ? false : true),
            ticks=legend_only ? nothing : :auto,
            framestyle=legend_only ? :none : :axes,
            showaxis=legend_only ? false : true,
            margin=0Plots.PlotMeasures.mm,
            left_margin=no_title ? 6Plots.PlotMeasures.mm : 2Plots.PlotMeasures.mm,
            bottom_margin=no_title ? 6Plots.PlotMeasures.mm : 2Plots.PlotMeasures.mm,
            top_margin=2Plots.PlotMeasures.mm,
            right_margin=2Plots.PlotMeasures.mm,
            size=legend_only ? (400, 600) : (no_title ? (600, 600) : (420, 420)),
        )
    end

    for i in 1:(size_x + 1)
        hline!(p, [i - 0.5], lw=0.3, color=:gray90, label=false)
    end
    for j in 1:(size_y + 1)
        vline!(p, [j - 0.5], lw=0.3, color=:gray90, label=false)
    end

    if n_region < size_x * size_y
        outside_label_added = false
        for r in 1:size_x, c in 1:size_y
            if !((r, c) in mdp.valid_region)
                lbl = !outside_label_added ? "Outside region" : false
                outside_label_added = true
                plot!(p, [c - 0.5, c + 0.5, c + 0.5, c - 0.5, c - 0.5],
                         [r - 0.5, r - 0.5, r + 0.5, r + 0.5, r - 0.5],
                      seriestype=:shape, fillcolor=:gray85, fillalpha=0.6,
                      linewidth=0, label=lbl)
            end
        end
    end

    for (i, j) in mdp.obstacles
        if (i, j) in mdp.valid_region
            plot!(p, [j], [i], seriestype=:scatter, color=:black, marker=:rect, markersize=4, label=nothing)
        end
    end

    if !isempty(base_segments)
        for (idx, seg) in enumerate(base_segments)
            if !isempty(seg)
                x_seg = [j for (i, j) in seg]
                y_seg = [i for (i, j) in seg]
                plot!(p, x_seg, y_seg, color=:gray60, linewidth=1.0, linestyle=:dash,
                      label=idx==1 ? "Shortest path" : false)
            end
        end
    end

    if !isempty(segments) && !isempty(actions_taken)
        drill_colors = [:magenta, :crimson, :darkorange]
        drill_idx = 0

        for (idx, segment) in enumerate(segments)
            if !isempty(segment)
                x_seg = [j for (i, j) in segment]
                y_seg = [i for (i, j) in segment]
                a = actions_taken[idx]

                if a.type == GOTO_GOAL
                    plot!(p, x_seg, y_seg, color=:green4, linewidth=2.5, label="→ goal")
                else
                    drill_idx += 1
                    c = drill_colors[mod1(drill_idx, length(drill_colors))]
                    label_text = "drill$(a.target_idx) (explore)"
                    plot!(p, x_seg, y_seg, color=c, linewidth=2.5, label=label_text)
                end
            end
        end
    elseif !isempty(full_path)
        x_path = [j for (i, j) in full_path]
        y_path = [i for (i, j) in full_path]
        plot!(p, x_path, y_path, color=:magenta, linewidth=2, label="Path")
    end

    selected_set = Set(selected_idxs)
    unselected_label_added = false

    for (i, pos) in enumerate(candidates)
        if !(i in selected_set)
            label_text = !unselected_label_added ? "Candidate (not selected)" : false
            unselected_label_added = true
            scatter!(p, [pos[2]], [pos[1]], color=:gray60, marker=:circle, markersize=4,
                     markerstrokewidth=0.5, label=label_text, alpha=0.6)
            if !isempty(var_map) && !no_title
                std_val = sqrt(var_map[pos[1], pos[2]])
                annotate!(p, pos[2]+1.5, pos[1]+1.0,
                          text("±$(round(std_val, digits=3))", 4, :left, :gray50))
            end
            if !no_title
                annotate!(p, pos[2]-1.5, pos[1]-1.0,
                          text("c$i", 4, :right, :gray50))
            end
        end
    end

    for (drill_i, cand_idx) in enumerate(selected_idxs)
        pos = candidates[cand_idx]
        label_text = drill_i == 1 ? "Drill site" : false
        scatter!(p, [pos[2]], [pos[1]], color=:red, marker=:star5,
                 markersize=no_title ? 10 : 7,
                 markerstrokewidth=1.5, markerstrokecolor=:darkred, label=label_text)
        if !isempty(ice_map) && !no_title
            m = ice_map[pos[1], pos[2]]
            annotate!(p, pos[2]+1.5, pos[1]+1.5,
                      text("$(round(m, digits=2))", 5, :left, :red))
        end
        if !no_title
            annotate!(p, pos[2]-1.5, pos[1]-1.0,
                      text("d$drill_i", 5, :right, :darkred))
        end
    end

    for (i, pos) in enumerate(pois)
        lbl = i == 1 ? "POI" : false
        scatter!(p, [pos[2]], [pos[1]], color=:dodgerblue, marker=:diamond, markersize=7,
                 markerstrokewidth=1.5, markerstrokecolor=:navyblue, label=lbl)
    end

    scatter!(p, [start_pos[2]], [start_pos[1]], color=:green, marker=:star5, label="Start", markersize=9,
             markerstrokewidth=1.5, markerstrokecolor=:darkgreen)
    scatter!(p, [goal_pos[2]], [goal_pos[1]], color=:blue, marker=:star5, label="Goal", markersize=9,
             markerstrokewidth=1.5, markerstrokecolor=:darkblue)

    out = if VIPER_TRAJ_TIKZ_FLAG
        base, _ = splitext(output_filename)
        base * ".tikz"
    else
        output_filename
    end
    savefig(p, out)
    println("Trajectory plot saved to $out")
end
function plot_completion_histogram(rollout_times::Vector{Float64},
                                   time_budget::Float64;
                                   output_filename::String="risk_histogram.png")
    no_title = get(ENV, "VIPER_TRAJ_NOTITLE", "0") == "1"
    dpi_setting = parse(Int, get(ENV, "VIPER_TRAJ_DPI", "100"))

    n = length(rollout_times)
    success_pct = round(100 * count(t -> t <= time_budget, rollout_times) / n, digits=1)

    xmin = floor(Int, minimum(rollout_times))
    xmax = ceil(Int, max(maximum(rollout_times), time_budget))
    p = histogram(rollout_times,
        bins=25,
        xlabel="Completion time (hours)",
        ylabel="Number of rollouts",
        xticks=xmin:1:xmax,
        title=no_title ? "" : "Completion-time distribution ($n rollouts, $success_pct% under budget)",
        legend=false,
        color=:steelblue,
        linecolor=:white,
        linewidth=0.4,
        titlefontsize=8,
        guidefontsize=no_title ? 22 : 8,
        tickfontsize=no_title ? 20 : 7,
        dpi=dpi_setting,
        margin=0Plots.PlotMeasures.mm,
        left_margin=no_title ? 10Plots.PlotMeasures.mm : 4Plots.PlotMeasures.mm,
        bottom_margin=no_title ? 8Plots.PlotMeasures.mm : 4Plots.PlotMeasures.mm,
        top_margin=2Plots.PlotMeasures.mm,
        right_margin=2Plots.PlotMeasures.mm,
        size=no_title ? (600, 600) : (420, 420),
    )
    vline!(p, [time_budget], color=:red, linewidth=1.8, linestyle=:dash, label="")
    savefig(p, output_filename)
    println("Completion-time histogram saved to $output_filename")
    println("Success rate (under $time_budget hr budget): $success_pct% ($(count(t -> t <= time_budget, rollout_times)) / $n)")
end
