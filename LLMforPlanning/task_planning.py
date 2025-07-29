import openai
import re
from params import args
from openai import OpenAI
from LLMforPlanning.prompt_template import *

def LLM_task_planning(now, factories, job, num_factories):

    prompt = jsp_prompt()
    messages = prompt.get_message(now, factories, job, num_factories)
    for attempt in range(1000):
        try:
            client = OpenAI(api_key=args.api_key, base_url=args.base_url)

            response_cur = client.chat.completions.create(
                model=args.model,
                messages=messages,  # information about system and user
                stream=False,
                temperature=0,  # args.temperature Control the randomness of the generated content
            )
            break
        except Exception as e:
            if attempt >= 10:
                chunk_size = max(int(chunk_size / 2), 1)
                print("Current Chunk Size", chunk_size)
            print(f"Attempt {attempt + 1} failed with error: {e}")
    content = response_cur.choices[0].message.content
    data = json.loads(content)
    selected_factory = data.get("Factory", "")
    return selected_factory

def robust_json_parse(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        try:
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print("Regex-based fallback failed:", e)
    return None