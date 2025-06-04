import re

from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import Literal

SUBMIT_MARKER = "#TO_TEACHER#"
END_MARKER = "#END_CONVERSATION#"


def check_submit_condition(state):
    submit_marker_present = SUBMIT_MARKER in state.get("messages", [])[-1].content

    return "yes" if submit_marker_present else "no"


def check_finish_condition(state, session_turn) -> Literal["yes", "no"]:
    session_turn_limit = state.get("session_turn", 0) >= session_turn
    end_marker_present = END_MARKER in state.get("messages", [])[-1].content

    return "yes" if session_turn_limit or end_marker_present else "no"


def transform_message_types(state, current_agent):
    messages = state.get("messages", [])
    senders = state.get("senders", ["teacher"])

    for i in range(0, len(messages)):
        if senders[i] == current_agent:
            if not isinstance(messages[i], AIMessage):
                messages[i] = AIMessage(content=messages[i].content)
        else:
            if not isinstance(messages[i], HumanMessage):
                messages[i] = HumanMessage(content=messages[i].content)


def extract_human_contents(state):
    contents = []
    marker_pattern = re.compile(r"#TO_STUDENT_ALPHA#|#TO_STUDENT_BETA#|#TO_TEACHER#|#END_CONVERSATION#")
    empty_role_pattern = re.compile(r'^(?:system|student_alpha|student_beta|teacher):\s*$')

    msgs = [msg for msg in state.get("messages", [])]

    for idx, msg in enumerate(msgs):
        raw = msg.content

        if isinstance(raw, list):
            texts = [seg["text"] for seg in raw if seg.get("type") == "text"]
            txt = "".join(texts)
        else:
            txt = str(raw)

        if idx == 0:
            txt = "system: " + txt

        txt = marker_pattern.sub("", txt).strip()

        if re.search(r"[)\uFF09]$", txt):
            txt += "."

        if empty_role_pattern.match(txt):
            continue

        contents.append(txt)

    return "\n".join(contents)
