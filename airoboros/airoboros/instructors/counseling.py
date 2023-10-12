import asyncio
import json
import os
import random
import re
import numpy as np


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
    topics, probs = instructor.get_instructor_topics(config)
    # topic_index = random.randint(0, len(topics) - 1)
    topic_index = np.random.choice(len(topics), p=probs)

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
            # topic_index += 1
            # if topic_index >= len(topics):
            #     topic_index = 0
            topic_index = np.random.choice(len(topics), p=probs)
        topics_str = "\n".join(
            [
                f" * message {idx + 1} must be related to topic: {json.dumps(topic)}"
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
        included_topics = []
        response_prompt = """
You are a helpful and professional mental health therapist. Please respond to the patient's message below as if you were a genuine human, offering your suggestions and guidance. Your response should be detailed and considerate, addressing the patient's concerns step by step without relying on professional jargon or redirecting them to a mental health expert.

Response Requirements:

1. Avoid using phrases that emphasize your non-professional status or limitations.
2. DO NOT direct the patient to a mental health expert, counselor, therapist or other similar people as if you were a mental health professional.
3. Provide a varied response without repeating the same patterns.
4. Do not start the response with "I'm sorry", or any similar patterns.
5. Do not start the response with "I'm really", or any similar patterns.
6. Do not start the response with "It sounds like", or any similar patterns.
7. Do not start the response with "It seems like", or any similar patterns.
8. Do not start the response with "It", or any similar patterns.
9. Do not start the response with "I understand", or any similar patterns.
10. Do not start the response with "It can be", or any similar patterns.
11. Do not start the response with "I hear you", or any similar patterns.
Please make sure to follow the requirements above in your response. 
Additionally, don't start with greetings, acknowledging challenges, or expressing your understanding and empathy toward the patient's situation.
JUST OUTPUT YOUR GUIDANCE AND SUGGESTIONS!! 
\n
Patient's message: 
                          """
        for idx, instruction in enumerate(re.findall(
            r"(?:^|\n)Instruction \d+\. (.*?)(?:$|(?=\nInstruction \d+\. ))", response, re.DOTALL
        )):
            if not instruction.strip() or await instructor.is_too_similar(
                instruction, min_score=min_score
            ):
                continue
            instructions.append(instruction)
            included_topics.append(current_topics[idx])
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
                "topic": current_topics[idx],
                "response": response.strip(),
                "category": "counseling",
            }
            if instructor.instructor_counts["counseling"] >= target_count:
                break
