import json
import numpy as np
import torch
import sys
import types
import simpy
import traceback
# from LLMforReward.jspEnvironments.workshopllm import JobShop
from Environments.DJobShop3 import DJobShop

# factor_role_instrutor = f"ROLE INSTRUCTION: You are good at understanding job shop scheduling problems and writing python codes.\
# You should fully understand the provided task and describe the exact observation and action form in the current decision point. \
# Then, based on your understanding and the goal of the problem, analyze potential positive and negative behaviours or statuses that can be reflected in the observation and action.\
# Finally, write an evaluation function that returns factors evaluating the current status from different aspects. \
# Note:1. Do not use information you are not given! \
# 2. Focus on the most relevant evaluation factors and use information in observation as little as possible. \
# 3. The code should be as generic, complete and not contain omissions! \
# 4. Avoid dividing by zero!\
# 5. The input variable is in the form of (batch_size, m, dim), please return a list of several evaluation factor arrays, each in the form of (batch_size, 1). \
# 6. Do NOT use decorators like @torch.jit.script or any TorchScript-related APIs.\
# 7. You must respond with **only a valid JSON object**, formatted **strictly without any Markdown, code block, or explanation** and do not include triple backticks (```), 'json', or any text outside the JSON. \
# 8. Your output must start directly with the curly brace:\
# Please think step by step and must adhere to the following JSON format (just replace the contents inside the () with your answer):"+\
# "{\
# Understand: (give your thought about the task),\
# Analyze: (think step by step and analyze potential positive and negative behaviors or statuses that can be reflected in which part of the observation and action), \
# Functions: (a python function with the form of 'def evaluation_func(observation, action):\
#  ... return [a list of evaluation factor arrays]')\
# }"

# direct_role_instrutor = f"ROLE INSTRUCTION: You are good at understanding job shop scheduling and writing python codes.\
# You should fully understand the provided task and describe the exact observation and action form in the current decision point. \
# Then, based on your understanding and the goal of the problem, analyze potential positive and negative behaviours or statuses that can be reflected in the observation and actions.\
# Finally, write a reward function that returns reward evaluating the current status from different aspects. \
# Note:1. Do not use information you are not given! \
# 2. Focus on the most relevant evaluation factors and use information in observation as little as possible. \
# 3. The code should be as generic, complete and not contain omissions! \
# 4. Avoid dividing by zero!\
# 5. The input variable is in the form of (batch_size, m, dim), please return a one element list which includes reward array in the form of (batch_size, 1). \
# 6. Do NOT use decorators like @torch.jit.script or any TorchScript-related APIs.\
# 7. You must respond with **only a valid JSON object**, formatted **strictly without any Markdown, code block, or explanation** and do not include triple backticks (```), 'json', or any text outside the JSON. \
# 8. Your output must start directly with the curly brace:\
# Please think step by step and must adhere to the following JSON format (just replace the contents inside the () with your answer) :"+\
# "{\
# Understand: (give your thought about the task),\
# Analyze: (think step by step and analyze potential positive and negative behaviors or statuses that can be reflected in which part of the observation and action), \
# Functions: (a python function with the form of 'def evaluation_func(observation, action):\
#  ... return [reward]')\
# }"

factor_role_instrutor = f"ROLE INSTRUCTION: You are good at understanding job shop scheduling problems and writing python codes.\
You should fully understand the provided task and describe the exact observation and action form in the current decision point. \
Then, based on your understanding and the goal of the problem, analyze potential positive and negative behaviours or statuses that can be reflected in the observation and action.\
Finally, write an evaluation function that returns factors evaluating the current status from different aspects. \
Note:1. Do not use information you are not given! \
2. Focus on the most relevant evaluation factors and use information in observation as little as possible. \
3. The code should be as generic, complete and not contain omissions! \
4. Avoid dividing by zero! \
5. The input variable 'states' is a 3D tensor with shape (batch_size, num_factories * m, 20), representing the concatenated local observations from all factories; the input variable 'actions' is a 2D tensor with shape (batch_size, num_factories), where each entry indicates the index of the selected operation-machine pair or 0 if no action is taken; m means the number of machines in each factory. \
6. Please return a list of several evaluation factor arrays, each in the form of (batch_size, 1). \
7. Do NOT use decorators like @torch.jit.script or any TorchScript-related APIs. \
8. You must respond with **only a valid JSON object**, formatted **strictly without any Markdown, code block, or explanation** and do not include triple backticks (```), 'json', or any text outside the JSON. \
9. Your output must start directly with the curly brace: \
10. You must use PyTorch operations with 'dim' instead of 'axis', as this function will receive PyTorch tensors. \
11. Avoid all kinds of index out-of-bound errors! Always check index validity before indexing into the observation or action tensor. \
Please think step by step and must adhere to the following JSON format (just replace the contents inside the () with your answer):" + \
"{\
Understand: (give your thought about the task),\
Analyze: (think step by step and analyze potential positive and negative behaviors or statuses that can be reflected in which part of the observation and action), \
Functions: (a python function with the form of 'def evaluation_func(observation, action): \
 ... return [a list of evaluation factor arrays]')\
}"

direct_role_instrutor = f"ROLE INSTRUCTION: You are good at understanding job shop scheduling and writing python codes.\
You should fully understand the provided task and describe the exact observation and action form in the current decision point. \
Then, based on your understanding and the goal of the problem, analyze potential positive and negative behaviours or statuses that can be reflected in the observation and actions.\
Finally, write a reward function that returns reward evaluating the current status from different aspects. \
Note:1. Do not use information you are not given! \
2. Focus on the most relevant evaluation factors and use information in observation as little as possible. \
3. The code should be as generic, complete and not contain omissions! \
4. Avoid dividing by zero! \
5. The input variable 'states' is a 3D tensor with shape (batch_size, num_factories * m, 20), representing the concatenated local observations from all factories; the input variable 'actions' is a 2D tensor with shape (batch_size, num_factories), where each entry indicates the index of the selected operation-machine pair or 0 if no action is taken; m means the number of machines in each factory. \
6. Please return a one element list which includes reward array in the form of (batch_size, 1). \
7. Do NOT use decorators like @torch.jit.script or any TorchScript-related APIs. \
8. You must respond with **only a valid JSON object**, formatted **strictly without any Markdown, code block, or explanation** and do not include triple backticks (```), 'json', or any text outside the JSON. \
9. Your output must start directly with the curly brace: \
10. You must use PyTorch operations with 'dim' instead of 'axis', as this function will receive PyTorch tensors. \
11. Avoid all kinds of index out-of-bound errors! Always check index validity before indexing into the observation or action tensor. \
Please think step by step and must adhere to the following JSON format (just replace the contents inside the () with your answer):" + \
"{\
Understand: (give your thought about the task),\
Analyze: (think step by step and analyze potential positive and negative behaviors or statuses that can be reflected in which part of the observation and action), \
Functions: (a python function with the form of 'def evaluation_func(observation, action): \
 ... return [reward]')\
}"




# State_forms = {
#     "DJobShop": "The state is concatenated from the observations of all agents at each decision point. The observation of each agent (i.e., a machine) at each decision point is 12 dimension:\
#     0: now: The current time step of the job-shop environment (i.e., the time point when a new job arrives, a machine breaks down, or an operation is processed).\
#     1: num_jobs: The total number of jobs assigned to the factory where the machine is located.\
#     2: CRJ_avg: The average job completion rates at a decision point.\
#     3: CRJ_std: The standard deviation of job completion rates at a decision point.\
#     4: TR_avg: The average processing delay rate of jobs at a decision point.\
#     5: TR_std: The standard deviation of processing delay rates at a decision point.\
#     6: Tard_e: The estimated tardiness rate at a decision point.\
#     7: Tard_a: The estimated tardiness rate at a decision point.\
#     8: U_m: The workload of a machines at a decision point.\
#     9: workload: The workload of a machines at a decision point.\
#     10: avg_PT: The average processing time of all ready operations that can be processed by the machine at a decision point. A ready operation is an unprocessed operation whose predecessor operations have been completed at this decision point.\
#     11: std_PT: The standard deviation of processing time of all ready operations that can be processed by the machine at a decision point.\
#     ",
#     }

State_forms = {
    "DJobShop":
    r"The state is concatenated from the observations of all agents at each decision point. The observations of each agent \( S \in \mathbb{R}^{m \times 20} \) is a matrix representation at each decision point, where \( m \) denotes the maximum number of ready operations in the system at that time step. Each row \( s[i,:] \) corresponds to a specific ready operation, and encodes both operation-level and machine-level information relevant to decision making.\n\
\n\
    - \( S[i,:m+6] \): Each row represents one ready operation. A ready operation is an unprocessed operation whose all predecessor operations have been completed. If \( s[i,:] \) is a zero vector (i.e., all elements are 0), it indicates a **padding row** used to maintain a fixed input shape when the number of ready operations is less than \( m \).\n\
\n\
    - \( S[:, 0:m] \): A machine-operation processing time matrix. Each entry \( S[i,j] \) denotes the processing time of the \( i \)-th ready operation on machine \( j \). A value of 0 indicates that the operation cannot be processed on that machine. Each row represents one ready operation.\n\
\n\
    - \( S[:, m:m+6] \): Operation-level features for each ready operation:\n\
        0. Job ID to which the operation belongs.\n\
        1. Job weight (priority) for the operation.\n\
        2. Average processing time of this operation across all candidate machines.\n\
        3. Number of machines that can process this operation.\n\
        4. Number of remaining unprocessed operations in the same job.\n\
        5. Estimated tardiness of the job.\n\
\n\
    - \( S[:, m+6:] \): Machine and factory-level features relevant to the operation:\n\
        0. Current utilization of the candidate machine.\n\
        1. Current workload of each machine.\n\
        2. Earliest available time of each machine.\n\
        3. Factory ID to which the corresponding job has been assigned.\n\
\n\
    This unified representation captures both compatibility between ready operations and machines, as well as global scheduling context, enabling reinforcement learning agents to make informed and efficient scheduling decisions under dynamic job arrivals and machine disturbances."
}

Problem_descriptions = {
    "DJobShop": (
        "This is a **distributed, flexible job shop scheduling problem** (DFJSP) with **new job arrivals**, involving the following key components:\n\n"

        "0. **Problem Overview**:\n"
        "   - Multiple jobs arrive dynamically and need to be scheduled across **multiple factories**.\n"
        "   - Each factory contains a unique set of machines. Each job must be fully processed within a single factory.\n"
        "   - Each operation in a job can be assigned to **one of several candidate machines**, each with **different processing times**.\n\n"

        "1. **Scheduling Goal**:\n"
        "   - Learn a global scheduling strategy that **minimizes the makespan** across all jobs in all tasks.\n"

        "2. **Task Definition**:\n"
        "   - A task contains a set of jobs that must be scheduled jointly to optimize a objective.\n"

        "3. **Job Structure**:\n"
        "   - Each job consists of a **sequence of dependent operations**.\n"
        "   - Each job is associated with:\n"
        "     - **Arrival time**: when the job becomes available for processing\n"
        "     - **Due date**: the deadline by which the job should ideally be completed\n"
        "     - **Weight**: a scalar indicating the urgency of the job; higher weights imply higher priority\n"
        "   - These attributes affect the reward function and scheduling decisions.\n\n"

        "4. **Operation Characteristics**:\n"
        "   - Each operation is only ready after all its **preceding operations are completed**.\n"
        "   - Each operation can be processed by **a subset of machines** of a factory, each with a distinct processing time.\n\n"

        "5. **Machine Attributes**:\n"
        "   - Each machine can process only one operation at a time.\n"
        "   - Processing times vary per operation.\n\n"

        "6. **Factory Constraints**:\n"
        "   - Each factory has a set of machines and can independently process entire jobs.\n"
        "   - **All operations of a job must be executed within the same factory**.\n\n"

        "7. **Agent Policy Design**:\n"
        "To minimize global makespan, we propose using a **multi-agent reinforcement learning framework**, where **each factory is equipped with its own scheduling agents**. "
        "Each agent represents a machine in a factory and is responsible for scheduling operations within its local factory scope, while indirectly cooperating to minimize makespan (time to complete all jobs)."
        "These agents may coordinate or operate independently to make real-time decisions for operation assignments within their local factory scope, while indirectly cooperating to minimize makespan (time to complete all jobs)."
    )
}

Action_forms = {
    "DJobShop": (
        "The dimensions of the multi-agent action are equal to the number of factories in the system. "
        "Each dimension corresponds to a specific factory and represents the operation-machine assignment selected by the agent within that factory at the current decision point."
        "Specifically, the value in each dimension is an integer between 0 and 101, where a non-zero value indicates the index of the selected operation-machine pair to be processed and 0 means that the agent chooses not to process any operations at that decision point."
    )
}

# Action_forms = {
#     "DJobShop": (
#         "The dimensions of the action are equal to the number of machines in the factory. "
#         "Each dimension represents the operation selected by a specific machine (i.e., one agent) at a given decision point. That is, each agent independently selects one operation to execute."
#         "Specifically, the value in each dimension is an integer between 0 and 5001, representing the index of the selected operation to process. "
#         "Note1: actions from different agents may conflict (e.g., multiple machines selecting the same operation). "
#         "Note2: once two or more agents have the same action, it is considered that the cooperation of the agents in that state has failed. Note that if the actions of two agents are both 0, it does not mean that their actions conflict, because an action of 0 means that they both choose not to process any operations."
#         "Note3: Such conflicts should be penalized during evaluation."
#     )
# }

class Base_prompt(object):
    def __init__(self, DJobShop, factor=True) -> None:
        self.map_name = DJobShop
        self.task_description = ''
        self.state_form = ''
        self.role_instruction = ''
        self.factor = factor

    def get_message(self):

        message=[]
        message.append({'role': 'user', 'content': self.task_description+self.state_form+self.role_instruction})
        return message

    def factor_check(self, out_content):
        error_idx = -1
        error_content = 'There is an error in your previous answer. Error:'
        pass_check = True
        factor_nums = []
        for i in range(len(out_content)):
            try:
                func = json.loads(out_content[i])['Functions']
                func = func.encode().decode('unicode_escape')

                func_lines = func.splitlines()
                # print(func)
                namespace = {'np': np, 'torch': torch}

                temp_module_name = f'temp_module_{i}'
                temp_module = types.ModuleType(temp_module_name)
                temp_module.__dict__.update(namespace)

                # if "@torch.jit.script\n" in func:
                #     func = func.replace("@torch.jit.script\n", "")
                # exec(func, namespace)   # execute the function to get the evaluation factors
                # active_evaluation_func = namespace['evaluation_func']  # get the function named evaluation_func
                # evaluation_factors = active_evaluation_func(np.array(self.global_state), np.array(self.action))  # , np.array([self.obs]*2))  # *2 is for check the function whether can handle batch input

                code = compile(func, filename=temp_module_name, mode='exec')
                exec(code, temp_module.__dict__)   # execute the function to get the evaluation factors
                active_evaluation_func = temp_module.evaluation_func   # get the function named evaluation_func

                evaluation_factors = active_evaluation_func(torch.tensor(self.global_state), torch.tensor(self.action))
                # evaluation_factors = active_evaluation_func(self.global_state, self.action)

                # evaluation_factors = active_evaluation_func(np.array([np.concatenate(self.obs)]*2), np.array([self.action]*2))#, np.array([self.obs]*2))  # *2 is for check the function whether can handle batch input
                # evaluation_factors = active_evaluation_func(np.array([np.concatenate(self.obs)]*2), np.array([self.action]*2))#, np.array([self.obs]*2))  # *2 is for check the function whether can handle batch input
                # evaluation_factors = active_evaluation_func(np.array([self.obs]*2), np.array([self.action]*2))#, np.array([self.obs]*2))
                factor_num = len(evaluation_factors)
                factor_nums.append(factor_num)
                if not self.factor and len(evaluation_factors) > 1:
                    pass_check = False
                    error_idx = i
                    error_content = f'There is an error in your previous answer. Error: The output should be a list with only one element, i.e., rewards.'
                    # if 'The output should be a list with only one element, i.e., rewards.' not in error_content:
                    #     error_content = f'{error_content} The output should be a list with only one element, i.e., rewards.'
                for factor in evaluation_factors:
                    if len(factor.shape) != 2 or factor.shape[0] != 1 or factor.shape[1] != 1:    # 这里batch size为1
                        pass_check = False
                        error_idx = i
                        error_content = f'There is an error in your previous answer. Error: The shape of the output factors should be (batch_size, 1).'
                        # if 'The shape of the output factors should be (batch_size, 1)' not in error_content:
                        #     error_content = f'{error_content} The shape of the output factors should be (batch_size, 1).'
            except Exception as e:
                pass_check = False
                error_idx = i
                # error_content = f'There is an error in your previous answer. Error:{e.args}' # with state_example: {np.round(np.stack(self.obs, axis=0), 2)}

                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_list = traceback.extract_tb(exc_traceback)

                # 只保留当前模块内的 traceback（即来自 LLM 生成的函数）
                relevant_frames = [
                    (filename, lineno, func_name, text)
                    for filename, lineno, func_name, text in tb_list
                    if filename == temp_module_name
                ]

                if not relevant_frames:  # 如果没找到匹配的 traceback，说明是语法错误或 import 错误
                    error_content = f"{error_content} {exc_type.__name__}: {str(e)}"
                else:
                    error_content = 'There is an error in your previous answer. Error:'
                    for filename, lineno, func_name, text in relevant_frames:
                        # 使用 func_lines 定位原始代码行
                        line_text = func_lines[lineno - 1].strip() if lineno <= len(func_lines) else 'unknown'
                        error_content += f" Line {lineno} '{line_text}': {exc_type.__name__}: {str(e)}; "

        return pass_check, error_idx, error_content, factor_nums

class jsp_prompt(Base_prompt):
    def __init__(self, args, dataSets, map_name='DJobShop',  factor=True) -> None:
        super().__init__(map_name)
        import gymnasium as gym
        # self.env = simpy.Environment()   #, exclude_current_positions_from_observation=False)
        self.env = DJobShop(simpy.Environment(), args)   #, exclude_current_positions_from_observation=False)
        self.env.reset(dataSets)
        self.obs = self.env.get_state(0, [], -1)
        self.global_state = np.concatenate([self.obs]*self.env.num_factories)
        self.global_state = np.expand_dims(self.global_state, 0)
        self.action = np.zeros((1, self.env.num_factories))
        # self.action = np.expand_dims(self.action, 0)
        self.task_description = f"TASK: {Problem_descriptions[map_name]}\n"
        self.state_form = f"OBSERVATION FORM: {State_forms[map_name]}\n"
        self.action_form = f"ACTION FORM: {Action_forms[map_name]}\n"
        self.role_instruction = factor_role_instrutor if factor else direct_role_instrutor