import asyncio
import json
import os
import random
import re


async def generate(instructor, **kwargs):
    """Generator for generic/counseling training data."""
    config = instructor.instructors.get("counseling")
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)
    if not target_count:
        return

    # Load the prompt template.
    path = config.get("prompt_path", "counseling.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # Load the topics.
    topics = instructor.get_instructor_topics(config)
    topic_index = random.randint(0, len(topics) - 1)

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score")
    if min_score is None:
        min_score = instructor.min_docsearch_score
    min_score = float(min_score)

    # Generate the instruction/response pairs until we reach the target count.
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    if "counseling" not in instructor.instructor_counts:
        instructor.instructor_counts["counseling"] = 0
    language = config.get("language") or instructor.language
    flesch = config.get("flesch") or instructor.default_flesch
    while instructor.instructor_counts["counseling"] < target_count:
        # Inject the topics to use for this batch.
        current_topics = []
        for _ in range(batch_size):
            current_topics.append(topics[topic_index])
            topic_index += 1
            if topic_index >= len(topics):
                topic_index = 0
        topics_str = "\n".join(
            [
                f" * instruction {idx + 1} must be related to topic: {json.dumps(topic)}"
                for idx, topic in enumerate(current_topics)
            ]
        )

        # Get a batch of instructions.
        prompt = template.format(
            batch_size=batch_size,
            topics=topics_str,
            topic_avoidance=instructor.topic_avoidance,
            language=language,
            flesch=flesch,
        )

        response = await instructor.generate_response(prompt, **api_params)
        if not response:
            continue

        # Parse instructions and generate responses.
        futures = []
        instructions = []
        response_prompt = """
                          Please respond to the patient's message as if you were a genuine human, without using phrases like "I'm not a professional" , "I can't replace the advice of a ... professional" or similar disclaimers that state you are not, can not or can't do anything. 
                          Do not include any professional help related content. 
                          Do not respond with the same pattern. 
                          Do not start with something similar to "I'm sorry to hear ... " or "It's understandable ..." or "I understand" or similar phrases that express your sympathy. 
                          Your response should be supportive and include detailed suggestions. 
                          Avoid directing the patient to a mental health professional, therapist, or expert as if you are roleplaying as a professional in the field related to the topics mentioned in the message, and demonstrate genuine care for the patient's feeling and well-being. 
                          Remember this is a conversation between you and the patient.
                          \n
                          Patient's message: 
                          """
        for instruction in re.findall(
            r"(?:^|\n)Instruction \d+\. (.*?)(?:$|(?=\nInstruction \d+\. ))", response, re.DOTALL
        ):
            if not instruction.strip() or await instructor.is_too_similar(
                instruction, min_score=min_score
            ):
                continue
            instructions.append(instruction)
            futures.append(
                instructor.generate_response(
                    response_prompt+instruction, messages=kwargs.get("messages", []), **api_params
                )
            )
        if not futures:
            continue
        responses = await asyncio.gather(*futures)
        for idx in range(len(futures)):
            response = responses[idx]
            if not response or not response.strip():
                continue
            yield {
                "instruction": instructions[idx].strip(),
                "response": response.strip(),
                "category": "counseling",
            }
            if instructor.instructor_counts["counseling"] >= target_count:
                break
