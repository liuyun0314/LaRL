from typing import Tuple, Dict
import math
import torch
from torch import Tensor
import random
import numpy as np
import simpy
import copy
import re
import argparse
from Environments.Task import Task
from Environments.Job import Job
from Environments.Operation import Operation
from Environments.Machine import Machine
from Environments.factory import Factory
from Environments.Reward import *
from Runner.utils import *
import ast
from LLMforPlanning.task_planning import LLM_task_planning as order_planning

class DJobShop:
    def __init__(self, env, args):
        # job shop
        self.env = env    # simpy environment
        self.decision_points = []  # the decision points of the job shop system
        self.dynamic_contor = 0  # the dynamic contor of the job shop system
        self.num_tasks = 0    # the number of tasks in the job shop system
        self.num_new_job = args.num_new_jobs    # the number of new jobs in each job
        self.num_jobs = self.num_new_job  # the total number of jobs in the job shop system
        self.num_machines = args.num_machines    # the number of machines in a factory
        self.num_factories = args.num_factories    # the number of factories in the job shop system
        self.act_dim = args.act_dim    # the dimension of the action space
        self.arrival_times = []
        self.next_new_job_AT = 0
        self.index_job = 0  # the index of the jobs that arrives in the job shop system
        self.in_system_job_num = 0
        self.in_system_job_dic = {}  # {'arrival_time': self.in_system_job_num}
        self.tasks_list = []
        self.jobs = []
        self.all_jobs = []
        self.new_jobs = []
        self.factories = []
        self.failure_machine = None
        self.done = False
        self.num_finished = 0
        self.span = 0
        self.completed_jobs = {}
        # self.reward_model = LLMRewardDecomposer(args)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)
        self.reward_scaling.reset()
        self.state_norm = Normalization(shape=args.input_obs_dim)

    def reset(self, dataSet):
        self.jobs = []
        self.all_jobs = []
        self.new_jobs = []
        self.index_job = 0  # the index of the jobs that arrives in the job shop system
        self.in_system_job_dic = {}  # {'arrival_time': self.in_system_job_num}
        self.failure_machine = None
        self.done = False
        self.num_finished = 0
        self.span = 0
        self.factories = []
        self.new_jobs = self.load_data(dataSet)  
        self.Factories_Initialization()
        for i in range(self.num_tasks):
            self.completed_jobs[i] = []
        self.next_new_job_AT = self.arrival_times[0]

        self.env = simpy.Environment()
        self.decision_points = self.arrival_times
        self.decision_points = sorted(self.decision_points)
        if self.decision_points[0] - self.env.now < 0:
            print("Error: the first decision point is earlier than the current time")
        self.env.timeout(self.decision_points[0] - self.env.now)
        self.env.run()

    def agent_obervation(self, f_id, ready_opera, llm_flag):
        obs = self.get_observation(f_id, ready_opera, llm_flag)
        state = np.concatenate(obs)
        obs = np.array(obs)
        return obs, state

    def agent_state(self, f_id, ready_opera, llm_flag):
        obs = self.get_state(f_id, ready_opera, llm_flag)
        state = np.concatenate(obs)
        obs = np.array(obs)
        return obs, state

    def load_data(self, dataSets):
        job_index = 0
        jobs = []
        arrival_time = dataSets[-2].strip().split()
        self.arrival_times = [float(time) for time in arrival_time]
        DDLs = dataSets[-1].strip().split()
        obj_str = dataSets[1].strip()
        info = dataSets[0].strip().split()
        self.num_tasks = int(info[0])
        self.num_jobs = int(info[1])
        self.num_machines = int(info[2])
        self.seed = int(info[3])
        self.num_factories = int(info[4])
        objectives = ast.literal_eval(obj_str)
        self.Task_Generator(objectives)
        for line in dataSets[3:-2]:
            AT = self.arrival_times[job_index]
            job = self.Job_Generator(line, AT, job_index)
            jobs.append(job)
            job_index += 1
        return jobs

    def Factories_Initialization(self):  # generate the factory list
        for f in range(self.num_factories):
            mac_list = []
            for m in range(self.num_machines):
                machine = Machine(m, f)
                mac_list.append(machine)
            factory = Factory(f, mac_list)
            self.factories.append(factory)

    def factory_reset(self):
        for f in self.factories:
            f.assigned_jobs = []
            f.idle_ratios = 0.0
            f.avg_UR = 0.0
            f.finished_jobs = []
            self.machins_reset(f.machines)

    def machins_reset(self, machines):
        for m in machines:
            m.currentTime = 0
            m.assignedOpera = []
            m.state = 'idle'
            m.busyTime = 0
            m.available = True

    def Task_Generator(self, objectives):    # generate the task
        for i in range(self.num_tasks):
            task = Task(i, objectives[i])
            self.tasks_list.append(task)

    def Job_Generator(self, line, arrival_time, globale_id):
        line = line.strip().split()
        task_id = int(line[0])
        job_id = int(line[1])
        weight = int(line[2])
        num_operations = int(line[3])
        operations = []
        counter = 4
        for i in range(num_operations):
            num_candidate_macs = int(line[counter])
            num_bits = 2 * num_candidate_macs
            lines = line[counter + 1:counter + num_bits+1]
            id_operation = i
            operation = self.Operation_Generator(id_operation, job_id, globale_id, task_id, arrival_time, lines)
            operations.append(operation)
            counter += (num_bits+1)
        job = Job(task_id, self.tasks_list[task_id].job_counter, weight, arrival_time, operations, globale_id)
        self.tasks_list[task_id].job_counter += 1
        return job

    def Operation_Generator(self, id_operation, id_job, globale_id, taskID, arrival_time, lines):  # generate the operation of a job

        candidate_machines = {}
        for i in range(0, len(lines), 2):
            machine_name = 'M' + lines[i]
            candidate_machines[machine_name] = int(lines[i+1])
        operation = Operation(id_operation, id_job, globale_id, taskID, candidate_machines, arrival_time)
        return operation

    def new_job_arrival(self):  # there a new job arrives and generate a new job
        if self.index_job < self.num_new_job:
            if self.env.now == self.next_new_job_AT:
                job = self.new_jobs[0]
                # print(f"Job {job.global_id} arrives at time {self.env.now}")
                self.jobs.append(job)
                self.index_job += 1
                self.record_jobs_arrival(job)
                self.tasks_list[job.idTask].jobsList.append(job)
                self.new_jobs.pop(0)
                self.dynamic_contor += 1
                if self.index_job < self.num_new_job:
                    self.next_new_job_AT = self.arrival_times[self.index_job]
                    if self.next_new_job_AT not in self.decision_points:
                        self.decision_points.append(self.next_new_job_AT)
                    if self.next_new_job_AT == self.env.now:  
                        self.new_job_arrival()
                if self.env.now not in self.decision_points:
                    self.decision_points.append(self.env.now)
                self.decision_points = sorted(self.decision_points)

    def record_jobs_arrival(self, job):  # record the arrival job
        self.in_system_job_num += 1
        if self.env.now not in self.in_system_job_dic:
            self.in_system_job_dic[self.env.now] = []
        self.in_system_job_dic[self.env.now].append(job)

    def get_state(self, f_id, ready_opera, llm_flag):

        if len(ready_opera) == 0:
            return np.zeros((self.factories[f_id].num_machines, 20))
        else:
            ready_OPs = ready_opera

            Tcur = []
            for m in self.factories[f_id].machines:
                Tcur.append(m.currentTime)

            ready_O_M = np.zeros((self.factories[f_id].num_machines, self.factories[f_id].num_machines))
            ready_op_features = np.zeros((self.factories[f_id].num_machines, 6))
            i = 0
            for op in ready_OPs:
                ready_op_features[i, :] = self.get_readyOpera_features(op, max(Tcur))
                for j, mac in enumerate(self.factories[f_id].machines):
                    if mac.name in op.cMachines:
                        ready_O_M[i][j] = op.cMachines[mac.name]

            ava_M = np.zeros((self.factories[f_id].num_machines, 4))
            for i in range(self.factories[f_id].num_machines):
                mac_state = self.get_mac_features(self.factories[f_id].machines[i], llm_flag[f_id])
                ava_M[i, :] = mac_state
            state = np.concatenate((ready_O_M, ready_op_features, ava_M), axis=1)
            return state

    def get_mac_features(self, m, llm_flag):
        U_m = 0
        workload = 0
        for o in m.assignedOpera:
            workload += o.duration
        if workload > 0:
            U_m = workload / m.currentTime
        mac_state = [U_m, workload, m.currentTime, llm_flag]
        return np.array(mac_state)

    def get_readyOpera_features(self, opera, T_cure):

        # the information of jobs
        Adjacent_edges_num = len(opera.cMachines)
        job_ID = opera.global_jobID
        job = self.jobs[job_ID]
        remian_oper_num = len(job.operation_list) - job.operation_list.index(opera)
        pts = list(opera.cMachines.values())
        mean_pt = sum(pts) / len(pts)

        C_j = 0  # The completion time of the last scheduled operation of job J_i until decision point
        ETL_i = 0
        for index, o in enumerate(job.operation_list):
            if o.completed:
                C_j = o.endTime
            else:
                cMachines = o.cMachines
                total_sum = sum(cMachines.values())
                mean_PT = total_sum / float(len(cMachines))
                ETL_i += mean_PT
        TR_j = (max(T_cure, C_j) + ETL_i - job.DT) / (C_j + ETL_i)

        return np.array([job_ID, job.weight, mean_pt, Adjacent_edges_num, remian_oper_num, TR_j])


    def get_observation(self, f_id, ready_opera, llm_flag, next_time=None):
        '''Return all agent observations in a list'''

        current_time = next_time if next_time is not None else self.env.now
        num_assigned_jobs = len(self.factories[f_id].assigned_jobs)
        if current_time not in self.arrival_times:
            llm_flag[f_id] = -1
        obs = [current_time, llm_flag[f_id], num_assigned_jobs]

        # avail_actions, ready_operas = self.get_avail_action(self.factories[f_id].machines, f_id, ready_opera)
        task_features = self.get_task_features(self.factories[f_id].assigned_jobs, self.factories[f_id].machines)
        obs.extend(task_features)
        observations = []
        for m in self.factories[f_id].machines:
            obs_i = []
            obs_i.extend(obs)
            mac_state = self.get_machine_features(ready_opera, m)
            obs_i.extend(mac_state)
            obs_i = self.state_norm(obs_i)
            observations.append(obs_i)

        return observations
        # return observations, avail_actions, ready_operas

    def get_task_features(self, jobs, machines):

        # the information of jobs
        CRJs = []  # the completion rate of jobs
        TR = []   #
        CKs = []
        for m in machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs) / self.num_machines
        if len(jobs) == 0:
            task_routing_state = [0, 0, 0, 0, 0, 0]
            return task_routing_state
        for j in jobs:
            OP_j = 0  # current operation number that has been completed of job J_i
            ETL_i = 0  # estimated completion time of the remaining operations of job J_i
            C_j = 0 # The completion time of the last scheduled operation of job J_i until decision point
            for index, o in enumerate(j.operation_list):
                if o.completed:
                    OP_j += 1
                    C_j = o.endTime
                else:
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    ETL_i += mean_PT
            CRJ = OP_j / len(j.operation_list)
            CRJs.append(CRJ)
            TR_j = (max(T_cure, C_j) + ETL_i - j.DT) / (C_j + ETL_i)   # the job processing delay rate
            TR.append(TR_j)
        CRJ_avg = np.mean(CRJs)  # 1. mean completion rate of jobs
        CRJ_std = np.std(CRJs)   # 2. standard deviation of completion rate of jobs
        TR_avg = np.mean(TR)  # 3. mean processing delay rate of jobs
        TR_std = np.std(TR)  # 4. standard deviation of processing delay rate of jobs

        CTs = []
        for m in machines:
            CTs.append(m.currentTime)
        T_cure = np.mean(CTs) / self.num_machines
        min_CT = min(CTs)

        N_tard, N_left = 0, 0
        N_Aleaft = 0
        for J in jobs:
            T_left = 0
            j = 0
            op_J = 0  # the number of operations that have been completed
            C_last = 999999  # the completion time of the last scheduled operation of job J_i until decision point
            for o in J.operation_list:
                if o.completed:
                    op_J += 1
                    C_last = o.endTime
                else:
                    N_left += 1  # The number of operations that have not been completed
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    T_left += mean_PT
                if T_left + max(T_cure, C_last) > J.DT:
                    j += 1
            N_tard += j
            if max(C_last, min_CT) > J.DT:
                N_Aleaft += len(J.operation_list) - op_J
        try:
            Tard_e = N_tard / N_left   # 5. Estimated tardiness rate Tard_e
        except:
            Tard_e = 999999

        try:
            Tard_a = N_Aleaft / N_left   # 6. Actual tardiness rate Tard_a
        except:
            Tard_a = 999999

        task_routing_state = [CRJ_avg, CRJ_std, TR_avg, TR_std, Tard_e, Tard_a]
        return task_routing_state

    def get_machine_features(self, ready_operas, m):
        U_m = 0
        busy = 0
        avail = 0
        workload = 0
        for o in m.assignedOpera:
            workload += o.duration
        if workload > 0:
            U_m = workload / m.currentTime

        matched_operas = []
        for op in ready_operas:
            if op is not None:
                belong_job_id = op.jobID
                belong_job = self.jobs[belong_job_id]
                factoyr_id = m.located_factory
                if belong_job in self.factories[factoyr_id].assigned_jobs:
                    if m.name in op.cMachines:
                        matched_operas.append(op.cMachines[m.name])
        if len(matched_operas) != 0:
            avg_PT = np.mean(matched_operas)
            std_PT = np.std(matched_operas)
        else:
            avg_PT = 0
            std_PT = 0
        mac_state = [U_m, workload, avg_PT, std_PT]
        return mac_state

    def get_ava(self, machines, ready_opera):
        avail = np.zeros((self.act_dim))
        for i, op in enumerate(ready_opera):
            begin_index = i * len(machines)
            if op is not None:
                for m_name, pt in op.cMachines.items():
                    match = re.search(r'M(\d+)', m_name)
                    m_id = int(match.group(1))
                    if machines[m_id].available and machines[m_id].currentTime <= self.env.now:  # the machine is available
                        avail[begin_index+m_id+1] = 1  
        if np.all(avail == 0):
            avail[0] = 1
        return avail

    def get_avail_action(self, machines, factoyr_id, ready_opera):
        num_machines = len(machines)
        avail = np.zeros((num_machines, self.act_dim))
        if len(ready_opera) == 0:
            avail[:, 0] = 1
        else:
            for j in range(num_machines):
                if machines[j].available:  # the machine is available
                    if machines[j].currentTime > self.env.now:  # the machine is busy
                        avail[j][0] = 1
                    else:
                        i = 1
                        for o in ready_opera:
                            if o is not None:
                                belong_job_id = o.jobID
                                belong_job = self.jobs[belong_job_id]
                                if belong_job in self.factories[factoyr_id].assigned_jobs:
                                    if machines[j].name in o.cMachines.keys():
                                        avail[j][i] = 1
                            i += 1
                        if sum(avail[j][:]) == 0:
                            avail[j][0] = 1

        return np.array(avail)

    def get_ready_opera(self, next_time=None, jobs=None):   # all available operations that can be assigned at this time

        current_time = next_time if next_time is not None else self.env.now
        jobs = jobs if jobs is not None else self.jobs
        counter = 0
        ready_opera = []
        ready_o = [None] * self.num_jobs
        if len(jobs) != 0:
            finish_counter = 0
            for j in jobs:
                pre_opera = j.operation_list[0]
                for o in j.operation_list:
                    if not o.completed:
                        if o.idOpertion == 0:
                            ready_o[counter] = o
                        else:
                            if pre_opera.endTime <= current_time:
                                ready_o[counter] = o
                        break
                    pre_opera = o
                counter += 1
        ready_opera = ready_o
        return ready_opera

    def get_routing_features(self, agent_id, machines):

        # the information of jobs
        CRJs = []  # the completion rate of jobs
        TR = []   #
        jobs = self.tasks_list[agent_id].jobsList
        CKs = []
        for m in machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs) / self.num_machines
        if len(jobs) == 0:
            task_routing_state = [0, 0, 0, 0, 0, 0]
            return task_routing_state
        for j in jobs:
            OP_j = 0  # current operation number that has been completed of job J_i
            ETL_i = 0  # estimated completion time of the remaining operations of job J_i
            C_j = 0 # The completion time of the last scheduled operation of job J_i until decision point
            for index, o in enumerate(j.operation_list):
                if o.completed:
                    OP_j += 1
                    C_j = o.endTime
                else:
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    ETL_i += mean_PT
            CRJ = OP_j / len(j.operation_list)
            CRJs.append(CRJ)
            TR_j = (max(T_cure, C_j) + ETL_i - j.DT) / (C_j + ETL_i)   # the job processing delay rate
            TR.append(TR_j)
        CRJ_avg = np.mean(CRJs)  # 1. mean completion rate of jobs
        CRJ_std = np.std(CRJs)   # 2. standard deviation of completion rate of jobs
        TR_avg = np.mean(TR)  # 3. mean processing delay rate of jobs
        TR_std = np.std(TR)  # 4. standard deviation of processing delay rate of jobs

        CTs = []
        for m in machines:
            CTs.append(m.currentTime)
        T_cure = np.mean(CTs) / self.num_machines
        min_CT = min(CTs)

        N_tard, N_left = 0, 0
        N_Aleaft = 0
        for J in jobs:
            T_left = 0
            j = 0
            op_J = 0  # the number of operations that have been completed
            C_last = 999999  # the completion time of the last scheduled operation of job J_i until decision point
            for o in J.operation_list:
                if o.completed:
                    op_J += 1
                    C_last = o.endTime
                else:
                    N_left += 1  # The number of operations that have not been completed
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    T_left += mean_PT
                if T_left + max(T_cure, C_last) > J.DT:
                    j += 1
            N_tard += j
            if max(C_last, min_CT) > J.DT:
                N_Aleaft += len(J.operation_list) - op_J
        try:
            Tard_e = N_tard / N_left   # 5. Estimated tardiness rate Tard_e
        except:
            Tard_e = 999999

        try:
            Tard_a = N_Aleaft / N_left   # 6. Actual tardiness rate Tard_a
        except:
            Tard_a = 999999

        task_routing_state = [CRJ_avg, CRJ_std, TR_avg, TR_std, Tard_e, Tard_a]
        return task_routing_state

    def act(self, all_assigments, factory_ready_operations, llm_flag, next_time=None):
        next_time = next_time if next_time is not None else self.env.now
        next_obs, next_state, next_ava, next_ready_operas = [], [], [], []
        for select_act, factoyr_id in zip(all_assigments, range(self.num_factories)):
            machines = self.factories[factoyr_id].machines
            ready_operas = factory_ready_operations[factoyr_id]
            if select_act == 0:
                continue
            else:
                select_oper_index = int((select_act - 1) / len(machines))
                opera = ready_operas[select_oper_index]
                mac_index = (select_act-1) % len(machines)
                mac = machines[mac_index]
                opera.assigned = True
                opera.completed = True
                opera.assignedMachine = mac.name
                opera.startTime = self.env.now
                opera.duration = opera.cMachines[mac.name]
                opera.endTime = self.env.now + opera.duration
                mac.currentTime = opera.endTime
                mac.busyTime += opera.duration
                mac.assignedOpera.append(opera)
                J = self.jobs[opera.global_jobID]
                if opera.idOpertion == 0:
                    J.RT = opera.startTime
                    J.waitTime = J.RT - J.AT
                if opera.idOpertion == len(J.operation_list) - 1:
                    J.completed = True
                    self.num_finished += 1
                    J.endTime = J.getEndTime()
                    self.completed_jobs[J.idTask].append(J)

                if mac.currentTime not in self.decision_points:
                    self.decision_points.append(mac.currentTime)
                    self.decision_points = sorted(self.decision_points)

                if self.num_finished == self.num_jobs:
                    self.done = True
                    for j in range(self.num_tasks):
                        self.tasks_list[j].completed = True
                        if len(self.completed_jobs[j]) > 0:
                            last_J = self.completed_jobs[j][-1]
                            self.tasks_list[j].endTime = last_J.endTime
                        else:
                            self.tasks_list[j].endTime = 0
                ready_operas.remove(opera)
        if all(len(sub) == 0 for sub in factory_ready_operations):
            self.trainsation()
        else:
            if all(x == 0 for x in all_assigments):  
                self.trainsation()

        for i in range(self.num_factories):
            obs, state = self.agent_state(i, factory_ready_operations[i], llm_flag)
            next_obs.append(obs)
            next_state.append(state)
        if self.env.now not in self.decision_points:
            self.decision_points.append(self.env.now)
        self.decision_points = sorted(self.decision_points)
        return next_obs, next_state, factory_ready_operations

    def trainsation(self):
        if len(self.decision_points) == 0:
            self.env.timeout(1)
            self.env.run()
        else:
            self.env.timeout(self.decision_points[0] - self.env.now)
            self.env.run()
        # self.new_job_arrival()
        if self.env.now not in self.decision_points:
            self.decision_points.append(self.env.now)
        if len(self.decision_points) == 0:
            self.decision_points.append(self.env.now)
        self.decision_points = sorted(self.decision_points)

    def assigned_factory(self):
        llm_flag = [-1 for _ in range(self.num_factories)] 
        selected_factory = -1
        if self.env.now in self.in_system_job_dic:
            new_jobs = self.in_system_job_dic[self.env.now]
            if len(new_jobs):
                llm_flag = [0 for _ in range(self.num_factories)]
                for job in new_jobs:
                    selected_factory = order_planning(self.env.now, self.factories, job, self.num_factories)
                    # selected_factory = AR(self.factories)  # select factory with minimum available time
                    job.assigned_factory = selected_factory
                    self.factories[selected_factory].assigned_jobs.append(job)
                    llm_flag[selected_factory] = 1
        return llm_flag, selected_factory

    def next_jobs(self, next_time):
        if next_time not in self.decision_points:  
            return self.jobs
        else:
            jobs = []
            jobs.extend(self.jobs)
            jobs.append(self.new_jobs[0])
            return jobs

    def step(self, all_assigments, ready_operas, llm_flag, next_time=None):  # step function for multi-agent transformer

        next_time = next_time if next_time is not None else self.env.now
        next_obs, next_state, next_ava, next_ready_operas = [], [], [], []
        for assigments, factoyr_id in zip(all_assigments, range(self.num_factories)):
            machines = self.factories[factoyr_id].machines
            for i, mac in enumerate(machines):
                select_act = assigments[0][i][0]
                if select_act == 0 or not mac.available:   # means the machine is not available or the agent does not select any operation
                    continue
                else:
                    opera = ready_operas[select_act-1]  
                    if opera is not None:
                        opera.assigned = True
                        opera.completed = True
                        opera.assignedMachine = mac.name
                        opera.startTime = self.env.now
                        opera.duration = opera.cMachines[mac.name]
                        opera.endTime = self.env.now + opera.duration
                        mac.currentTime = opera.endTime
                        mac.busyTime += opera.duration
                        mac.assignedOpera.append(opera)
                        J = self.jobs[select_act-1]
                        if opera.idOpertion == 0:
                            J.RT = opera.startTime
                            J.waitTime = J.RT - J.AT
                        if opera.idOpertion == len(J.operation_list) - 1:
                            J.completed = True
                            self.num_finished += 1
                            J.endTime = J.getEndTime()
                            self.completed_jobs[J.idTask].append(J)

                        if mac.currentTime not in self.decision_points:
                            self.decision_points.append(mac.currentTime)
                            self.decision_points = sorted(self.decision_points)

                        if self.num_finished == self.num_jobs:
                            self.done = True
                            for i in range(self.num_tasks):
                                self.tasks_list[i].completed = True
                                if len(self.completed_jobs[i]) > 0:
                                    last_J = self.completed_jobs[i][-1]
                                    self.tasks_list[i].endTime = last_J.endTime
                                else:
                                    self.tasks_list[i].endTime = 0
            obs = self.get_observation(factoyr_id, ready_operas, llm_flag, next_time)
            state = np.concatenate(obs)
            next_obs.append(obs)
            next_state.append(state)
        return next_obs, next_state

    def makespan(self):
        max_makespan = 0
        for job in self.jobs:
            makespan = self.estimated_end_time(job)
            if makespan > max_makespan:
                max_makespan = makespan
        return max_makespan

    def interval_time(self, all_C_last, actions):
        flattened_actions = actions.flatten().astype(int)
        num_agents = len(flattened_actions)
        # rewards = np.zeros(num_agents)

        now_C_last = C_last(self.tasks_list)
        interval_time = []
        for i in range(self.num_tasks):
            for now, old, job in zip(now_C_last[i], all_C_last[i], self.tasks_list[i].jobsList):
                interval = job.weight * (now - old)
                interval_time.append(interval)
        avg_interval_time = -np.mean(interval_time)
        rew = np.exp(avg_interval_time)

        non_zero_actions = flattened_actions[flattened_actions != 0] 
        unique_actions, counts = np.unique(non_zero_actions, return_counts=True)
        conflict_actions = unique_actions[counts > 1]
        penalty = np.zeros(num_agents)
        for a in conflict_actions:
            mask = (flattened_actions == a).nonzero().squeeze()
            k = len(mask)
            if k > 1:
                penalty[mask] -= 0.5 * k ** 2

        rewards = rew + np.mean(penalty)
        return rewards, {'rew': rew, 'conflict_penalty': penalty}

    def WT_mean_func(self, jobs):
        '''
        calculate the mean weighted tardiness of jobs
        :param jobs:
        :return:
        '''
        WTs = []
        for j in jobs:
            WT = j.weight * max(0, j.endTime - j.DT)
            WTs.append(WT)
        return np.mean(WTs)

    def WT_max_func(self, jobs):
        '''
        calculate the maximum weighted tardiness of jobs
        :param jobs:
        :return:
        '''
        WTs = []
        for j in jobs:
            WT = j.weight * max(0, j.endTime - j.DT)
            WTs.append(WT)
        return max(WTs)

    def WF_mean_func(self, jobs):
        '''
        calculate the weighted flow time of jobs
        :param jobs: a job sets
        :return:
        '''
        WFs = []
        for j in jobs:
            WF = j.weight * max(0, j.endTime - j.RT)
            WFs.append(WF)
        return np.mean(WFs)

    def estimated_end_time(self, j, machines):
        '''
        calculate the estimated end time of job j
        :param j:
        :return:
        '''
        CKs = []
        for m in machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs)
        WFs = []
        if not j.completed:
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            C_last = j.RT + j.span
            Cmax = max(T_cure, C_last) + sum_t_avg
        else:
            Cmax = j.endTime
        return Cmax

    #########################################################################
    def get_obs(self, factory_id, ready_opera, llm_flag):
        features = []
        jobs = self.factories[factory_id].assigned_jobs
        machines = self.factories[factory_id].machines
        if llm_flag == factory_id:   # 1. whether the new job is assigned to the assigned factory, 1 means yes, 0 means no,-1 means no new job
            whether_in = 1
        else:
            if llm_flag != -1:
                whether_in = 0
            else:
                whether_in = -1
        features.append(whether_in)
        assigned_machine = factory_id  # 2. the assigned factory id of the ready operations
        features.append(assigned_machine)
        ready_PT, ava = self.global_readyOpera_PT(ready_opera, machines)  # 3. processing time of the ready operations
        features.extend(ready_PT)
        mac_fateure = self.mac_feats(machines)   # 4. features of the machines at current time
        features.extend(mac_fateure)
        job_features = self.get_task_features(jobs, machines)  # 5. features of the jobs at current time
        features.extend(job_features)
        return np.array(features), ava

    def mac_feats(self, machines):
        fets = []
        for m in machines:
            fets.append(m.currentTime)   # next avaliable time
            workload = 0
            for o in m.assignedOpera:
                workload += o.duration
            fets.append(workload)   # workload of the machine
        return fets

    def global_readyOpera_PT(self, opera, machines):
        ava = np.zeros(len(machines))
        ready_PT = []  # 3. processing time of the ready operations
        for i, m in enumerate(machines):
            if m.name in opera.cMachines:
                ready_PT.append(opera.cMachines[m.name])
                ava[i] = 1
            else:
                ready_PT.append(0)
                ava[i] = 0
        return ready_PT, ava

def AR(factories):
    all_available_time = []
    for i, factory in enumerate(factories):
        available_time = factory.available_mac_time()
        min_pt = float('inf')
        for item in available_time:
            pt = list(item.values())[0]
            if pt < min_pt:
                min_pt = pt
        all_available_time.append(min_pt)
    selected_factory = np.argmin(all_available_time)
    return selected_factory

if __name__ == '__main__':
    from params import args
    file_path = '/GenertedInstances/trainingData/100/10125.txt'
    with open(file_path, 'r') as f:
        dataSets = f.readlines()
    env = simpy.Environment()
    envs = DJobShop(env, args)
    envs.reset(dataSets)
    envs.agent_obervation(0)
    print('Done')