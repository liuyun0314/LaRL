import hydra   
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
import openai
import re
from openai import OpenAI
import subprocess
from pathlib import Path
import shutil
import time
from creat_scene import create_scene, make_env

from track_Info import *
from utils import file_to_string, get_function_signature, load_tensorboard_logs
import MERL

ROOT_DIR = os.getcwd()
JSP_ROOT_DIR = f"{ROOT_DIR}/../Environments"
TRAIN_ROOT_DIR = '/used_trains'

@hydra.main(config_path='cfg', config_name='config')
def main(cfg):
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

    env_name = cfg.env.env_name.lower()
    env_parent = 'workShop'
    class_task_file = '/Environments/Task.py'
    class_job_file = '/Environments/Job.py'
    class_operation_file = '/Environments/Operation.py'
    class_machine_file = '/Environments/Machine.py'
    task_file = f'/Environments/{env_parent}.py'
    task_obs_file = f'/Environments/{env_parent}_obs.py'
    task_objectve_file = f'/Environments/{env_parent}_objectives.py'
    class_task_code_string = file_to_string(class_task_file)
    class_job_code_string = file_to_string(class_job_file)
    class_operation_code_string = file_to_string(class_operation_file)
    class_machine_code_string = file_to_string(class_machine_file)
    task_code_string = file_to_string(task_file)
    task_obs_string = file_to_string(task_obs_file)
    task_objective_string = file_to_string(task_objectve_file)
    output_file = f"{ROOT_DIR}/jspEnvironments/{env_name}{suffix.lower()}.py"

    # Loading all prompts
    prompt_dir = f"{ROOT_DIR}/prompt"
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_properties=class_task_code_string, job_properties=class_job_code_string, operation_properties=class_operation_code_string, machine_properties=class_machine_code_string,
                     task_reward_signature_string=reward_signature, Calculate_task_objectives=task_objective_string, observation_calculation=task_obs_string, jobShop=task_code_string) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_string, task_description=problem_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
    task_code_string = task_code_string.replace(problem_name, problem_name + suffix)

    # Create Task YAML files
    Scene_yaml_file = '/LLMforReward/cfg'

    # Initialize parameters
    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None

    # reward generation loop
    for iter in range(cfg.iteration):
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = 1

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        ################### used for generating reward function ###################
        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
                    response_cur = client.chat.completions.create(
                        model=cfg.model,
                        messages=messages,   # information about system and user
                        extra_body={"enable_thinking":True},
                        stream=False,
                        temperature=cfg.temperature,  # Control the randomness of the generated content
                        n=chunk_size  # The number of generated results
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Failed to generate response and Code terminated due to too many failed attempts, skipping...")
                exit()
            # responses inculdes id, object, created, model, yin, yang, message, choices, usage
            responses.extend(response_cur.choices)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: LLM Output:\n " + responses[0].message.content + "\n")
        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        code_runs = []
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id].message.content
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""', 
                r'""(.*?)""',  
                r'"(.*?)"',  
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()  # Obtain code
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            processed_lines = []
            found_def, get_signature_flag = False, True
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def ") and "__init__" not in line and get_signature_flag:
                    found_def = True
                    get_signature_flag = False
                    gpt_reward_signature = line.strip()
                    match = re.search(r'\(.*\)', line)
                    if match:
                        input_lst = match.group(0)
                if found_def:
                    processed_lines.append(line)
            code_string = "\n".join(processed_lines)

            code_runs.append(code_string)  # save the generated code
            reward_signature = [
                f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",  # Reward function signature
                f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
            ]

            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])   # add indent with eight Spaces to each line

            if "@torch.jit.script" not in code_string:
                code_string = "    @torch.jit.script\n" + code_string

            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + code_string)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + code_string)
            else:
                raise NotImplementedError

            # Save the new environment code with the reward function that contains valid code string
            with open(output_file, 'w') as file:
                # import necessary packeges
                file.writelines("from typing import Tuple, Dict" + '\n')
                file.writelines("import math" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')
                file.writelines(task_code_string_iter + '\n')
                # if "@torch.jit.script" not in code_string:
                #     code_string = "@torch.jit.script\n" + code_string
                # file.writelines(code_string + '\n')

            with open(f"/LLMforReward/jspEnvironments/response_code/env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"{TRAIN_ROOT_DIR}/response/env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()

            ################### used for feedback ###################
            rl_filepath = f"{used_ROOT_DIR}/used_trains/training_log/env_iter{iter}_response{response_id}.txt"  # record the log in trining process
            # os.makedirs(rl_filepath, exist_ok=True)
            MERL.trainer()
            with open(rl_filepath, 'w') as f: 
                process = subprocess.Popen(['/anaconda3/bin/python', '-u', f'{used_ROOT_DIR}/MERL.py'],
                                           stdout=f, stderr=f)
                block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)  # Check if the training has processed successfully
                rl_runs.append(process)

        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []

        exec_success = False

        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()   # Run the core of the training algorithm by starting a subprocess
            rl_filepath = f"{TRAIN_ROOT_DIR}/training_log/env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"{TRAIN_ROOT_DIR}/response/env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read()    # read the log file and check if there are any error message
            except:
                content = execution_error_feedback.format(
                traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)   # if there are not erros, return '', else return error message
            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):  
                        break   # check wether the tensorboard directory is in the last line of the output
                tensorboard_logdir = line.split(':')[-1].strip()  # spit the line according to the ':' and get a list
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)   # a dictionary of tensorboard logs, which includes scalar data in training process
                max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]   # get the maximum number of decision steps in training process
                epoch_freq = max(int(max_iterations // 10), 1)
                content += policy_feedback.format(epoch_freq=epoch_freq)

            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)
            content += code_output_tip
            contents.append(content)

            # Repeat the iteration if all code generation failed
            if not exec_success and cfg.sample != 1:
                execute_rates.append(0.)
                max_successes.append(DUMMY_FAILURE)
                max_successes_reward_correlation.append(DUMMY_FAILURE)
                best_code_paths.append(None)
                logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
                continue

        if cfg.sample == 1:
            best_sample_idx = 0
            best_content = contents[best_sample_idx]
        else:
            best_sample_idx = np.argmax(np.array(successes))
            best_content = contents[best_sample_idx]

        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx].message.content}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx].message.content}
            messages[-1] = {"role": "user", "content": best_content}


if __name__ == '__main__':
    main()
