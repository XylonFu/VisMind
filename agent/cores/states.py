from langgraph.graph import add_messages
from typing_extensions import Annotated, List, TypedDict


class StudentsTeacherState(TypedDict):
    messages: Annotated[List, add_messages]
    session_turn: int
    senders: list
