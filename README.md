# Large Language Model Explainability for Planning Lunar Rover Trajectories under Uncertainty
By Michelle Ho, Maximilian Adang, Zachary Booth, Evan Astle, Margaret Hansen, Edward Balaban, and Mykel J. Kochenderfer

Planning planetary rover trajectories under prob-
abilistic uncertainty is a multi-objective optimization problem
that science teams currently iteratively solve and verify by
hand. This process becomes impractical when timelines and
constraints frequently shift throughout the mission planning
phase. Sampling-based planners such as Monte Carlo Tree
Search can speed up this process, but their outputs do not
preserve traceability to the science objectives that justify each
plan, which the science team needs to verify. We propose an
interactive assistant based on a large language model (LLM)
that can explain planner trajectories using domain vocabulary
and re-run the planner under user-specified conditions to com-
pare alternatives. Previous work has used LLMs inside planner
decision-making loops, but we explore using them for post-
planning explanation, where they are more likely to be trusted
since they do not directly influence planner decisions. Our
framework pairs a simulator inspired by the NASA Volatiles
Investigating Polar Exploration Rover (VIPER) mission with a
ReAct queryable interface that calls curated action tools and
retrieval functions. We evaluate the assistant on a query bank
consisting of plan comparisons, feasibility, and lookup queries.
Experiments show that 90.9% of the assistant’s responses are
faithful to the planner’s outputs, a nearly 65% improvement
over zero-shot prompting of the same LLM without tool access,
demonstrating the potential for trustworthy explainability of
uncertainty-aware traverses with an LLM-based system.
