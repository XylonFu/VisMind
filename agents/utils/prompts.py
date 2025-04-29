from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_student_alpha_prompt():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "Role: student_alpha\n"
                "Task: Contribute to a multi-turn conversation with student_beta to solve a geometry problem. Incorporate any provided images and text, and strictly follow the given problem context.\n\n"
                "Instructions:\n"
                "1. Develop a detailed discussion where you articulate your reasoning, think aloud, and challenge student_beta’s approaches.\n"
                "2. Incorporate reflection, debate, and evidence-based challenges to encourage error detection and collaborative correction.\n"
                "3. Seamlessly continue the conversation by integrating any hints or feedback from the teacher.\n"
                "4. Only output your conversation contribution, excluding any extra content, praise, thanks, or encouragement.\n\n"
                "START THE CONVERSATION NOW. BEGIN WITH 'student_alpha:'."
            )
        ),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt


def get_student_beta_prompt():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "Role: student_beta\n"
                "Task: Contribute to a multi-turn conversation with student_alpha to solve a geometry problem. Incorporate any provided images and text, and strictly follow the given problem context.\n\n"
                "Instructions:\n"
                "1. Develop a detailed discussion where you articulate your reasoning, think aloud, and challenge student_alpha’s approaches.\n"
                "2. Incorporate reflection, debate, and evidence-based challenges to encourage error detection and collaborative correction.\n"
                "3. Seamlessly continue the conversation by integrating any hints or feedback from the teacher.\n"
                "4. After reasoning, place '#TO_TEACHER#' at the end if you believe everything is correct, or '#TO_STUDENT_ALPHA#' if you find mistakes.\n"
                "5. Only output your conversation contribution, excluding any extra content, praise, thanks, or encouragement.\n\n"
                "START THE CONVERSATION NOW. BEGIN WITH 'student_beta:'."
            )
        ),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt


def get_teacher_prompt(question, solution):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "Role: teacher\n"
                "Task: Evaluate the conversation between student_alpha and student_beta based on the geometry problem, images, and text provided. You have the following reference:\n"
                "Question: " + question + "\n"
                "Solution: " + solution + "\n\n"
                "Instructions:\n"
                "1. Review the conversation for both the reasoning process and the final answer accuracy.\n"
                "2. Identify any errors, misconceptions, or incomplete reasoning in the discussion.\n"
                "3. Provide constructive feedback, hints, and suggestions that guide the students to refine the conversation without revealing the solution.\n"
                "4. After reviewing, place '#END_CONVERSATION#' at the end if everything is correct, or '#TO_STUDENT_BETA#' if you find mistakes.\n"
                "5. Only output your feedback contribution, excluding any extra content, praise, thanks, or encouragement.\n\n"
                "PROVIDE YOUR FEEDBACK NOW. BEGIN WITH 'teacher:'."
            )
        ),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt


def get_generator_prompt():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "Role: Conversation Generator\n"
                "Task: Generate a multi-turn conversation between two students solving a math geometry problem. The discussion should incorporate any provided images and text, and strictly adhere to the given problem context.\n\n"
                "Instructions:\n"
                "1. Develop a detailed discussion where both students articulate their reasoning, think aloud, and challenge each other’s approaches.\n"
                "2. Incorporate reflection, debate, and evidence-based challenges to encourage error detection and collaborative correction.\n"
                "3. Seamlessly continue the conversation by integrating any hints or feedback from the Conversation Supervisor.\n"
                "4. Emphasize the process and reasoning without directly providing the final answer.\n"
                "5. Output only the conversation; do not include any additional content.\n\n"
                "Generate the conversation now."
            )
        ),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt


def get_supervisor_prompt(question, solution):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                "Role: Conversation Supervisor\n"
                "Task: Evaluate the conversation generated by the Conversation Generator based on the math geometry problem, accompanying images, and text. You have the following reference:\n"
                "Question: " + question + "\n"
                "Solution: " + solution + "\n\n"
                "Instructions:\n"
                "1. Review the conversation for both the reasoning process and the final answer accuracy.\n"
                "2. Identify any errors, misconceptions, or incomplete reasoning in the discussion.\n"
                "3. Provide constructive feedback, hints, and suggestions that guide the generator to refine the conversation without revealing the solution.\n"
                "4. If the conversation is fully correct in both process and outcome, output '#END#' at the end to signal completion.\n"
                "5. Please don't output any '#END#' if any mistakes are found.\n\n"
                "Provide your feedback now."
            )
        ),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt
