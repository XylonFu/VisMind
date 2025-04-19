from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.cores.nodes import generator_node, supervisor_node
from agent.cores.states import StudentsTeacherState
from agent.utils.helpers import check_finish_condition


def graph(generator_config, supervisor_config, graph_config):
    session_turn = graph_config.get("session_turn")
    generator_prompt = generator_config.pop("prompt")
    supervisor_prompt = supervisor_config.pop("prompt")

    generator_model = ChatOpenAI(**generator_config)
    supervisor_model = ChatOpenAI(**supervisor_config)

    generator = generator_prompt | generator_model
    supervisor = supervisor_prompt | supervisor_model

    graph_builder = StateGraph(StudentsTeacherState)
    graph_builder.add_node("generator", lambda state: generator_node(state, generator))
    graph_builder.add_node("supervisor", lambda state: supervisor_node(state, supervisor))

    graph_builder.add_edge("generator", "supervisor")
    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: check_finish_condition(state, session_turn),
        {"yes": END, "no": "generator"}
    )

    graph_builder.set_entry_point("generator")

    return graph_builder.compile()
