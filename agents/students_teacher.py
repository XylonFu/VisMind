from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from agent.cores.nodes import student_alpha_node, teacher_node, student_beta_node
from agent.cores.states import StudentsTeacherState
from agent.utils.helpers import check_finish_condition, check_submit_condition


def graph(student_alpha_config, student_beta_config, teacher_config, graph_config):
    session_turn = graph_config.get("session_turn")
    student_alpha_prompt = student_alpha_config.pop("prompt")
    student_beta_prompt = student_beta_config.pop("prompt")
    teacher_prompt = teacher_config.pop("prompt")

    student_alpha_model = ChatOpenAI(**student_alpha_config)
    student_beta_model = ChatOpenAI(**student_beta_config)
    teacher_model = ChatOpenAI(**teacher_config)

    student_alpha = student_alpha_prompt | student_alpha_model
    student_beta = student_beta_prompt | student_beta_model
    teacher = teacher_prompt | teacher_model

    graph_builder = StateGraph(StudentsTeacherState)
    graph_builder.add_node("student_alpha", lambda state: student_alpha_node(state, student_alpha))
    graph_builder.add_node("student_beta", lambda state: student_beta_node(state, student_beta))
    graph_builder.add_node("teacher", lambda state: teacher_node(state, teacher))

    graph_builder.add_edge("student_alpha", "student_beta")
    graph_builder.add_conditional_edges(
        "student_beta",
        lambda state: check_submit_condition(state),
        {"yes": "teacher", "no": "student_alpha"}
    )
    graph_builder.add_conditional_edges(
        "teacher",
        lambda state: check_finish_condition(state, session_turn),
        {"yes": END, "no": "student_beta"}
    )

    graph_builder.set_entry_point("student_alpha")
    return graph_builder.compile()
