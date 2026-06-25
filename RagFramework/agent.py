import os
import dspy
from RagFramework.tools import (
    run_simulation,
    run_scenario,
    set_mission_start,
    evaluate_poi,
    evaluate_poi_list,
    replan_with_map,
    run_arbitrary_region,
    select_custom_sites,
    change_order,
    risk_assessment,
    show_plot,
    read_source_code,
    read_data_file,
    search_mission_document,
    list_available_files,
)


PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'system_prompt.txt')
try:
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read()
except Exception:
    SYSTEM_PROMPT = ""


class VIPERSignature(dspy.Signature):
    __doc__ = SYSTEM_PROMPT
    user_query: str = dspy.InputField(desc="The user's question or request")
    response: str = dspy.OutputField(desc="Helpful response")


ACTION_TOOLS = [
    run_simulation, run_scenario, set_mission_start, evaluate_poi,
    evaluate_poi_list, replan_with_map, run_arbitrary_region,
    select_custom_sites, change_order, risk_assessment,
]
RETRIEVAL_TOOLS = [
    show_plot, read_source_code, read_data_file,
    search_mission_document, list_available_files,
]


def build_agent(model: str = None, api_key: str = None):
    if model is None:
        try:
            from RagFramework.utils import config
            model = f"openai/{config.OPENAI_MODEL}"
        except Exception:
            model = "openai/gpt-4o"

    if api_key is None:
        try:
            from RagFramework.utils import config
            api_key = config.OPENAI_API_KEY
        except Exception:
            pass

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    lm = dspy.LM(model=model, cache=False, temperature=0.7)
    dspy.configure(lm=lm)

    tools = [dspy.Tool(fn) for fn in ACTION_TOOLS + RETRIEVAL_TOOLS]
    return dspy.ReAct(VIPERSignature, tools=tools, max_iters=7)


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def ask(user_query: str) -> str:
    agent = get_agent()
    result = agent(user_query=user_query.strip())
    return result.response
