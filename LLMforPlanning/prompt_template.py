import json
import numpy as np
import torch
import sys
import simpy

role_instrutor = (
    "You are an intelligent planner in a multi-factory scheduling system. "
    "Your task is to assign a newly arrived job to one of several factories, based on the job and factory attributes. "
    "You should first analyze the situation in natural language (briefly), and then choose the best factory. "

    "Your output must be **a single valid JSON object**, with exactly two keys:\n"
    "1. 'Analysis': a brief explanation of your reasoning (string)\n"
    "2. 'Factory': the selected factory ID (integer)\n\n"

    "Important rules:\n"
    "- Your output must start directly with the curly brace '{'\n"
    "- Do NOT include any Markdown, code blocks, or extra text\n"
    "- Do NOT write 'json', 'Answer:', or anything before or after the JSON\n"

    "Example format:\n"
    "{\"Analysis\": \"Factory 1 has the most idle machines and shortest delay.\", \"Factory\": 1}"
)


def get_newJob_attributes(task_ID, weight, arrival_time, due_date, expected_time, num_operations):
    new_job_attributes = (
        "The newly arrived job has the following attributes:\
        0: task_ID (int): The ID of the task to which this new arrival job belongs is {ID}.\
        1: weight, due  (int): the weight of the job is {weight}, e.g.,(0: normal, 1: urgency).\
        1: arrival_time (float): the time the job arrived in the system is {arrival_time}.\
        2: due_date (float): the time the job is due to be completed is {due_date}.\
        3: expected_time (float): the expected processing time of the job is {expected_time}.\
        4: num_operations (int): the number of operations required to complete the job is {num_operations}.\
        "
    ).format(ID=task_ID, weight=weight, arrival_time=arrival_time, due_date=due_date, expected_time=expected_time,
             num_operations=num_operations)
    return new_job_attributes

def get_factory_attributes(factory_id, num_machines, avg_util, assigned_jobs, available_time, ratio, avg_edelay):
    factory_attributes = (
        "The {factory_id}-th factory has the following attributes:\n"
        "0: id (int): The ID of the factory is {factory_id}.\n"
        "1: number_machines (int): the number of machines in the factory is {num_machines}.\n"
        "2: average_utilization (float): the average utilization of the machines in the factory is {avg_util}.\n"
        "3: assigned_jobs (int): the number of jobs currently assigned to the factory is {assigned_jobs}.\n"
        "4: earliest_start_time (list): the earliest available time of each machine in this factory is {available_time}.\n"
        "5: idle_ratios (float): the current proportion of idle machines is {ratio}.\n"
        "6: estimated_delay_ratio (float): the proportion of expected delayed jobs is {avg_edelay}.\n"
    ).format(factory_id=factory_id, num_machines=num_machines, avg_util=avg_util, assigned_jobs=assigned_jobs, available_time=available_time, ratio=ratio, avg_edelay=avg_edelay)
    return factory_attributes

problem_description = (
    "You are tasked with making job-to-factory assignment decisions in a multi-factory production environment. "
    "New jobs arrive dynamically over time, and each job must be assigned to one of several factories for processing. "
    "Each job is characterized by attributes such as task ID, weight, arrival time, due date, expected processing time, and the number of required operations. "
    "Each factory has identical machine configuration, but may differ in real-time attributes like machine utilization and the number of jobs currently assigned. "
    "The goal is to assign each newly arrived job to the most appropriate factory, balancing the workload, avoiding overload, and considering potential delay risks. "
    "You should make decisions based on the provided attributes of the new job and the current status of each factory."
)

def get_action_form(num_factories):
    Action_forms = (
        "The action is a **single integer** representing the ID of the selected factory to which the newly arrived job will be assigned for processing. "
        "The available factories are identified by their integer IDs, ranging from 0 to {num}-1, where {num} is the total number of factories. "
        "Note: 1. Avoid assigning jobs to overloaded factories, if such constraints are specified in the factory attributes (e.g., high utilization or delay risk)."
    ).format(num=num_factories)
    return Action_forms

class jsp_prompt():
    def __init__(self) -> None:
        self.task_description = ''
        self.state_form = ''
        self.action_form = ''
        self.task_description = f"TASK: {problem_description}\n"
        self.role_instruction = role_instrutor
    def get_prompt(self, now, factories, job, num_factories):
        factory_attributes = self.all_factory_attributes(now, factories)
        new_job_attributes = self.new_job_attributes(job)
        self.state_form = f"OBSERVATION FORM: {factory_attributes}\n + {new_job_attributes}\n"
        self.action_form = f"ACTION FORM: {get_action_form(num_factories)}\n"

    def get_message(self, now, factories, job, num_factories):
        message = []
        self.get_prompt(now, factories, job, num_factories)
        message.append({'role': 'user', 'content': self.task_description + self.state_form + self.action_form + self.role_instruction})
        return message

    def all_factory_attributes(self, now, factories):
        role_instrutor = (f"The following is the attributes of each factory in the current:\n")
        for factory_id, factory in enumerate(factories):
            avg_util = factory.get_UR(now)
            available_time = factory.available_mac_time()
            ratio = factory.get_idle_ratio(now)
            avg_edelay = factory.estimated_average_delay()
            factory_attribute = get_factory_attributes(factory_id, factory.num_machines, avg_util, len(factory.assigned_jobs),
                                                        available_time, ratio, avg_edelay)
            role_instrutor = role_instrutor + factory_attribute + '\n'
        return role_instrutor

    def new_job_attributes(self, job):
        task_ID = job.idTask
        weight = job.weight
        arrival_time = job.AT
        due_date = job.DT
        expected_time = job.estimatedCompletionTime
        num_operations = len(job.operation_list)
        new_job_attributes = get_newJob_attributes(task_ID, weight, arrival_time, due_date, expected_time, num_operations)
        return new_job_attributes

