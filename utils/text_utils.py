import json

from langchain_core.messages import AIMessage, HumanMessage


def process_answer(answer: str) -> str:
    return answer.split("故选")[0].strip()


class MessageEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (AIMessage, HumanMessage)):
            return {
                "type": o.__class__.__name__,
                "content": o.content,
                "id": getattr(o, "id", None),
                "additional_kwargs": getattr(o, "additional_kwargs", {}),
                "response_metadata": getattr(o, "response_metadata", {}),
                "usage_metadata": getattr(o, "usage_metadata", {}),
            }
        return super().default(o)
