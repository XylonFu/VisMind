from agents.utils.helpers import transform_message_types, extract_human_contents
from agents.utils.prompts import get_teacher_user_prompt


def student_alpha_node(state, student_alpha):
    transform_message_types(state, current_agent="student_alpha")

    result = student_alpha.invoke(state)
    session_turn = state.get("session_turn", 0)
    senders = state.get("senders", ["system"]) + ["student_alpha"]

    return {"messages": [result], "session_turn": session_turn, "senders": senders}


def student_beta_node(state, student_beta):
    transform_message_types(state, current_agent="student_beta")

    result = student_beta.invoke(state)
    session_turn = state.get("session_turn", 0) + 1
    senders = state.get("senders", ["system"]) + ["student_beta"]

    return {"messages": [result], "session_turn": session_turn, "senders": senders}


def teacher_node(state, teacher, reference):
    question = reference.get("question")
    solution = reference.get("solution")
    image = reference.get("image")

    transform_message_types(state, current_agent="teacher")
    conversation = extract_human_contents(state)
    prompt = get_teacher_user_prompt(conversation, question, solution, image)

    result = teacher.invoke(prompt)
    session_turn = state.get("session_turn", 0)
    senders = state.get("senders", ["system"]) + ["teacher"]

    return {"messages": [result], "session_turn": session_turn, "senders": senders}


def generator_node(state, generator):
    transform_message_types(state, current_agent="generator")

    result = generator.invoke(state)
    session_turn = state.get("session_turn", 0)
    senders = state.get("senders", ["system"]) + ["generator"]

    return {"messages": [result], "session_turn": session_turn, "senders": senders}


def supervisor_node(state, supervisor):
    transform_message_types(state, current_agent="supervisor")

    result = supervisor.invoke(state)
    session_turn = state.get("session_turn", 0) + 1
    senders = state.get("senders", ["system"]) + ["supervisor"]

    return {"messages": [result], "session_turn": session_turn, "senders": senders}
