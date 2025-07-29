import hydra   
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
import openai
import re
from openai import OpenAI
from LLMforReward.prompt.prompt_template import *
from LLMforReward.utils import file_to_string, get_function_signature, load_tensorboard_logs
from params import args
ROOT_DIR = os.getcwd()
JSP_ROOT_DIR = f"{ROOT_DIR}/../Environments"
TRAIN_ROOT_DIR = '/used_trains'

# @hydra.main(config_path='cfg', config_name='config')
def LLM_generate_reward(cfg, factor=True):
    workspace_dir = os.getcwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Config: {cfg}")

    openai.api_key = cfg.api_key

    problem_name = cfg.env.task
    problem_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"using LLM: {model}")
    logging.info(f"Problem: " + problem_name)
    logging.info(f"Problem description: " + problem_description)

    # Initialize parameters
    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    total_samples = 2
    chunk_size = 1
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = f"{root_dir}/jspEnvironments/"
    # output_file = f"{ROOT_DIR}/jspEnvironments/"
    with open(args.root_dir, 'r') as f:
        dataSet = f.readlines()
    prompt = jsp_prompt(args, dataSet, 'DJobShop', factor=factor)
    message = prompt.get_message()
    all_message = prompt.get_message()
    recheck_message = [{'role': 'user', 'content': "You have generated several evaluation functions. \
    Please summarize them and generate a new evaluation function that incorporates all the evaluation factors.\
    If there are other important evaluation factors, please include them as well.\
    Your final output must be a **valid JSON object**, and strictly adhere to the following JSON format (just replace the () with your answer and output exactly this JSON format. Nothing more):"+\
    "{\
    Understand: (give your thought about the task),\
    Summary: (think step by step and summarize initial evaluation function), \
    Functions: (The new python function with the form of 'def evaluation_func(observation, action):\
    ... return [a list of evaluation factor arrays]')\
    }"}]
    if total_samples == 1:
        recheck_message = []

    out_content = LLM_inference(cfg.model, message, cfg.api_key, cfg.base_url, cfg.temperature, total_samples, chunk_size)
    message.append({'role': 'assistant', 'content': str(out_content)})
    all_message.append({'role': 'assistant', 'content': str(out_content)})
    check_phases = len(recheck_message) if len(recheck_message) > 0 else 1
    for i in range(check_phases):
        if len(recheck_message) > 0:
            new_content = recheck_message.pop(0)['content']
            message.append({'role': 'user', 'content': new_content})
            all_message.append({'role': 'user', 'content': new_content})
            out_content = LLM_inference(cfg.model, message, cfg.api_key, cfg.base_url, cfg.check_temperature, total_samples, chunk_size)
            message.append({'role': 'assistant', 'content': str(out_content)})
            all_message.append({'role': 'assistant', 'content': str(out_content)})

        for recheck_count in range(10):
            pass_check, error_idx, error_content, factor_nums = prompt.factor_check(out_content)
            if pass_check:
                break
            if recheck_count == 0:
                message[-1] = {'role': 'assistant', 'content': out_content[error_idx]}
                message.append({'role': 'user', 'content': error_content})
                all_message.append({'role': 'user', 'content': error_content})
            else:
                message[-2] = {'role': 'assistant', 'content': out_content[error_idx]}
                message[-1] = {'role': 'user', 'content': error_content}
                all_message.append({'role': 'user', 'content': error_content})
            out_content = LLM_inference(cfg.model, message, cfg.api_key, cfg.base_url, cfg.check_temperature, total_samples, chunk_size)

        if pass_check:
            id = 6
            np.save(output_file+'response_code/'+f'response_{id}.npy', out_content)
            np.save(output_file+'dialog_code/'+f'dialog_{id}.npy', all_message)
            np.save(output_file+'factor_num/'+f'factor_num_{id}.npy', factor_nums)

def LLM_inference(model, messages, api_key, base_url, temperature, total_samples, chunk_size):
    responses = []
    for i in range(total_samples):
        for attempt in range(1000):
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)

                response_cur = client.chat.completions.create(
                    model=model,
                    messages=messages,  # information about system and user
                    stream=False,
                    temperature=temperature,  # Control the randomness of the generated content
                    n=chunk_size  # The number of generated results
                )
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                print(f"Attempt {attempt + 1} failed with error: {e}")

        content = response_cur.choices[0].message.content
        if content.startswith("```json"):
            data = json.loads(content)
            filtered_data = {
                "Understand": data.get("Understand", ""),
                "Summary": data.get("Summary", ""),
                "Functions": data.get("Functions", "")
            }
            content = json.dumps(filtered_data, indent=2, ensure_ascii=False)
        responses.append(content)
    return responses

def get_function(out_content):
    all_results = []

    for content in out_content:

        # 1. Extract the Understanding content
        understand_match = re.search(r"### Understand\s+(.*?)(?=\n### Analyze)", content, re.DOTALL)
        understand = understand_match.group(1).strip() if understand_match else ""

        # 2. Extract the Analysis content
        analyze_match = re.search(r"### Analyze\s+(.*?)(?=\n### Functions)", content, re.DOTALL)
        analyze = analyze_match.group(1).strip() if analyze_match else ""

        # 3. Extract the Function content
        function_matches = re.findall(r"```python(.*?)```", content, re.DOTALL)
        functions = "\n\n".join(f.strip() for f in function_matches)

        result = {
            "Understand": understand,
            "Analyze": analyze,
            "Functions": functions
        }
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        all_results.append(json_str)
    return all_results


if __name__ == '__main__':
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("cfg/config.yaml")
    LLM_generate_reward(cfg)
