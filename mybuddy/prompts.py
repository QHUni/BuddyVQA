import json


LOW_ACTIVE_SUMMARY = """You are an intelligent multimodal assistant tasked with summarizing egocentric video segments to extract long-term, stable, and reusable contextual information. Your goal is to produce a concise textual summary that captures information in long videos, including key activities, recurring patterns, background environment, salient objects, user routines, and consistent spatial layout.

Focus on information that remains stable or frequently appears across this time window. Identify recurring actions, repeated interactions, or common tasks performed by the user. Capture stable environmental structure such as room layout, tool locations, furniture arrangement, or habitual user workflows. Extract objects or entities that reliably appear or are frequently manipulated across the window.

Your summary should represent general knowledge useful for future reasoning, not for a specific question. Write clear, compact sentences and provide only the long-term summary text."""


QUESTION_FILTER = """You are an intelligent multimodal assistant designed to identify whether an incoming question in a streaming egocentric QA scenario is chained or unchained. Determine whether the question relies on earlier context and, if so, which prior QA pair is most relevant.

Classify the incoming question as either chained or unchained. A chained question may contain ego-deictic expressions or implicit references to past events, actions, or answers. If unchained, do not select a historical QA pair. If chained, select the single most relevant QA pair based on semantic relation, shared objects or actions, temporal continuity, or referential expressions. Focus on meaningful contextual grounding.

Return only a JSON object with keys type and base_qa. type must be chained or unchained. base_qa must be the qid of the selected pair or null.

Current Question: {question}
Historical QAs: {historical_qas}"""


VISUAL_RETRIEVAL = """You are an intelligent multimodal assistant designed to retrieve visual information from egocentric video memories for real-time question answering. Generate concise visual descriptions, filter long-term summaries for information relevant to the question, and detect eye-gaze points and hand bounding boxes that help ground deictic references.

Describe only visible, grounded content. Focus on objects, spatial relations, user interactions, and scene attributes relevant to the question. Select only intermediate or long-term memories that meaningfully support the question. For gaze and hand detections, return normalized coordinates between 0 and 1 and do not hallucinate.

Return only a JSON object with keys caption, filtered_memory, and coordinates. caption must describe the relevant active and semi-active frames. filtered_memory must be a string or list. coordinates must be a list of objects.

Incoming Question: {question}
Active frame timestamps: {active_timestamps}
Semi-active frame timestamps: {semi_timestamps}
Low-active summaries: {low_summaries}"""


FINAL_QA = """You are an intelligent multimodal assistant designed to answer egocentric, streaming questions using retrieved visual information, historical QA context, and embodied features. First rewrite the incoming question into one explicit, fully disambiguated question, resolving expressions such as this, that, it, here, or there. Then answer the rewritten question using only the supplied evidence. Keep the answer concise and do not hallucinate.

Return only a JSON object with keys rephrased_question and answer.

Incoming Question: {question}
Retrieved Visual Descriptions: {visual_descriptions}
Relevant Historical QA: {relevant_qa}
Embodied Feature Coordinates: {coordinates}"""


EVALUATION = """You are evaluating a generative answer for egocentric streaming video question answering. Compare the prediction with the correct answer set. Treat synonyms and paraphrases as valid only when they preserve the same grounded entity, event, temporal interpretation, and semantic content. For chained questions, require consistency with the intended contextual referent.

Return only a JSON object with keys pred and score. pred must be yes or no. score must be an integer from 0 to 5.

Question: {question}
Correct Answer Set: {answers}
Predicted Answer: {prediction}"""


def question_filter_prompt(question: str, historical_qas: list[dict]) -> str:
    return QUESTION_FILTER.format(question=question, historical_qas=json.dumps(historical_qas, ensure_ascii=False))


def visual_retrieval_prompt(question: str, active_timestamps: list[float], semi_timestamps: list[float], low_summaries: list[str]) -> str:
    return VISUAL_RETRIEVAL.format(question=question, active_timestamps=active_timestamps, semi_timestamps=semi_timestamps, low_summaries=json.dumps(low_summaries, ensure_ascii=False))


def final_qa_prompt(question: str, visual_descriptions: dict, relevant_qa: dict | None, coordinates: list) -> str:
    return FINAL_QA.format(question=question, visual_descriptions=json.dumps(visual_descriptions, ensure_ascii=False), relevant_qa=json.dumps(relevant_qa, ensure_ascii=False), coordinates=json.dumps(coordinates, ensure_ascii=False))


def evaluation_prompt(question: str, answers: list[str], prediction: str) -> str:
    return EVALUATION.format(question=question, answers=json.dumps(answers, ensure_ascii=False), prediction=prediction)

