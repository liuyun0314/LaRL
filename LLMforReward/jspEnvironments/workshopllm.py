from typing import Tuple, Dict
import math
import torch
from torch import Tensor
import random
import numpy as np
import simpy
import copy
import argparse
from Environments.Task import Task
from Environments.Job import Job
from Environments.Operation import Operation
from Environments.Machine import Machine
from Environments.factory import Factory
from Environments.Reward import *
import ast

class JobShop:
    def __init__(self, env, args):
        # job shop
        self.env = env    # simpy environment
        self.decision_points = [0]  # the decision points of the job shop system
        self.dynamic_contor = 0  # the dynamic contor of the job shop system
        self.num_tasks = 0    # the number of tasks in the job shop system
        self.num_new_job = args.num_new_jobs    # the number of new jobs in each job
        self.num_jobs = self.num_new_job  # the total number of jobs in the job shop system
        self.num_machines = args.num_machines    # the number of machines in the job shop system
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
        self.machines = []
        self.factories = []
        self.failure_machine = None
        self.done = False
        self.num_finished = 0
        self.span = 0
        self.completed_jobs = {}
        self.machine_queue = {}

    def reset(self, dataSet):
        self.index_job = 0  # the index of the jobs that arrives in the job shop system
        self.in_system_job_dic = {}  # {'arrival_time': self.in_system_job_num}
        self.failure_machine = None
        self.done = False
        self.num_finished = 0
        self.span = 0
        #  without warmup jobs
        self.jobs = []
        self.all_jobs = []

        self.Factories_Initialization()
        self.new_jobs = self.load_data(dataSet)
        for i in range(self.num_tasks):
            self.completed_jobs[i] = []
        self.Machine_Initializationr()
        for m in self.machines:
            self.machine_queue[m.name] = []
        self.next_new_job_AT = self.arrival_times[0]
        self.decision_points.append(self.arrival_times[0])
        obs, ava, ready_operas = self.get_observation()
        state = obs.copy()
        return obs, state, ava, ready_operas

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
        objectives = ast.literal_eval(obj_str)
        self.Task_Generator(objectives)
        for line in dataSets[3:-2]:
            AT = self.arrival_times[job_index]
            job = self.Job_Generator(line, AT, job_index)
            jobs.append(job)
            job_index += 1
        return jobs

    def Factories_Initialization(self):  # generate the factory list
        m_list = []
        for m in range(self.num_machines):
            machine = Machine(m, 0)
            m_list.append(machine)
        for f in range(self.num_factories):
            factory = Factory(f, m_list)
            self.factories.append(factory)


    def Machine_Initializationr(self):  # generate the machine list
        current_time = 0
        for m in range(self.num_machines):
            machine = Machine(m, current_time)  # if the current time is less than or equal to self.env.now, the machine is idle.
            self.machines.append(machine)

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

    def Job_Generator(self, line, arrival_time, job_index):
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
            operation = self.Operation_Generator(id_operation, job_id, task_id, arrival_time, lines)
            operations.append(operation)
            counter += (num_bits+1)
        job = Job(task_id, self.tasks_list[task_id].job_counter, weight, arrival_time, operations, job_index)
        self.tasks_list[task_id].job_counter += 1
        return job

    def Operation_Generator(self, id_operation, id_job, taskID, arrival_time, lines):  # generate the operation of a job

        candidate_machines = {}
        for i in range(0, len(lines), 2):
            machine_name = 'M' + lines[i]
            candidate_machines[machine_name] = int(lines[i+1])
        operation = Operation(id_operation, id_job, taskID, candidate_machines, arrival_time)
        return operation

    def new_job_arrival(self):  # there a new job arrives and generate a new job
        if self.index_job < self.num_new_job:
            if self.env.now == self.next_new_job_AT:
                job = self.new_jobs[0]
                self.jobs.append(job)
                self.index_job += 1
                self.record_jobs_arrival()
                self.tasks_list[job.idTask].jobsList.append(job)
                self.new_jobs.pop(0)
                self.dynamic_contor += 1
                if self.index_job < self.num_new_job:
                    self.next_new_job_AT = self.arrival_times[self.index_job]
                    if self.next_new_job_AT not in self.decision_points:
                        self.decision_points.append(self.next_new_job_AT)
                    if self.next_new_job_AT == self.env.now:  # 可能出现arrival_interval[self.index_job]=0的情况
                        self.new_job_arrival()
                if self.env.now not in self.decision_points:
                    self.decision_points.append(self.env.now)
                self.decision_points = sorted(self.decision_points)

    def record_jobs_arrival(self):  # record the arrival job
        self.in_system_job_num += 1
        self.in_system_job_dic[self.env.now] = self.in_system_job_num

    def get_observation(self):
        '''Return all agent observations in a list'''
        obs = [self.env.now, len(self.jobs), self.num_tasks]
        # The average level of the number of jobs in multiple tasks
        jobs_num_avg = np.mean([len(task.jobsList) for task in self.tasks_list])
        obs.append(jobs_num_avg)
        # Standard deviation of the number of jobs in multiple tasks
        jobs_num_std = np.std([len(task.jobsList) for task in self.tasks_list])
        obs.append(jobs_num_std)
        # Average remaining time and completion rate of multiple tasks
        residual_time = [None for i in range(self.num_tasks)]
        percentage_complete = [None for i in range(self.num_tasks)]
        for task in self.tasks_list:
            if len(task.jobsList) == 0:
                percentage_complete[task.idTask] = 999999
                residual_time[task.idTask] = 999999
            else:
                percentage_complete[task.idTask] = len(self.completed_jobs[task.idTask]) / len(task.jobsList)
                counter_time = 0
                for job in task.jobsList:
                    counter_time += (job.DT - self.env.now)
                residual_time[task.idTask] = counter_time / len(task.jobsList)
        obs.append(np.mean(residual_time))
        obs.append(np.std(residual_time))
        obs.append(np.mean(percentage_complete))
        obs.append(np.std(percentage_complete))

        task_features = self.get_task_features()
        obs.extend(task_features)
        observations = []
        for i in range(self.num_machines):
            obs_i = []
            obs_i.extend(obs)
            mac_state = self.get_machine_features(i)
            obs_i.extend(mac_state)
            # normalization
            normal_obs_i = [(x - np.min(obs_i))/(np.max(obs_i)-np.min(obs_i)) for x in obs_i]
            observations.append(normal_obs_i)
        avail_actions, ready_operas = self.get_avail_action()
        return observations, avail_actions, ready_operas

    def get_task_features(self):

        # the information of jobs
        CRJs = []  # the completion rate of jobs
        TR = []   #
        jobs = self.jobs
        CKs = []
        for m in self.machines:
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
        for m in self.machines:
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

    def get_machine_features(self, i):
        U_m = 0
        busy = 0
        avail = 0
        workload = 0
        m = self.machines[i]
        for o in m.assignedOpera:
            workload += o.duration
        if workload > 0:
            U_m = workload / m.currentTime
        if m.currentTime > self.env.now:  # the machine is busy
            busy = 1
        if m.available:
            avail = 1
        mac_state = [U_m, busy, avail, workload]
        return mac_state

    def get_avail_action(self):

        avail = np.zeros((self.num_machines, self.act_dim))
        ready_opera = self.get_ready_opera()
        if len(ready_opera) == 0:
            for j in range(self.num_machines):
                if self.machines[j].available:  # the machine is available
                    avail[j][0] = 1
        else:
            for j in range(self.num_machines):
                if self.machines[j].available:  # the machine is available
                    if self.machines[j].currentTime > self.env.now:  # the machine is busy
                        avail[j][0] = 1    # if the machine is busy, the machine cannot select any operation.
                    else:
                        i = 1
                        for o in ready_opera:
                            if o is not None:
                                if self.machines[j].name in o.cMachines.keys():
                                    avail[j][i] = 1
                            i = i + 1
                        if sum(avail[j][:]) == 0 and self.machines[j].available:
                            avail[j][0] = 1

        return np.array(avail), ready_opera

    def get_ready_opera(self):
        counter = 0
        ready_opera = []
        ready_o = [None] * self.num_jobs
        if len(self.jobs) != 0:
            finish_counter = 0
            for j in self.jobs:
                if j.operation_list[-1].completed and j.operation_list[-1].endTime <= self.env.now:
                    finish_counter = finish_counter + 1
                else:
                    pre_opera = j.operation_list[0]
                    for o in j.operation_list:
                        if not o.completed:
                            if o.idOpertion == 0:
                                ready_o[counter] = o
                            else:
                                if pre_opera.completed and pre_opera.endTime <= self.env.now:
                                    ready_o[counter] = o
                            break
                        pre_opera = o
                counter += 1
            if finish_counter == len(self.jobs):
                ready_opera = []
            else:
                ready_opera = ready_o
        all_none = all(item is None for item in ready_opera)
        if not all_none:
            ready_opera = ready_o
        return ready_opera

    def get_routing_features(self, agent_id):

        # the information of jobs
        CRJs = []  # the completion rate of jobs
        TR = []   #
        jobs = self.tasks_list[agent_id].jobsList
        CKs = []
        for m in self.machines:
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
        for m in self.machines:
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


    def step(self, assigments, ready_operas):  # step function for multi-agent transformer
        for i, mac in enumerate(self.machines):
            select_act = assigments[0][i][0]
            if select_act == 0 or not mac.available:   #  #  means the machine is not available or the agent does not select any operation
                continue
            else:
                opera = ready_operas[select_act-1]  #
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
                    J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
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
        obs, ava, ready_operas = self.get_observation()
        state = obs.copy()
        return obs, state, ava, ready_operas

    def interval_time2(self, all_C_last):
        now_C_last = C_last(self.tasks_list)
        interval_time = []
        for i in range(self.num_tasks):
            for now, old, job in zip(now_C_last[i], all_C_last[i], self.tasks_list[i].jobsList):
                interval = job.weight * (now - old)
                interval_time.append(interval)
        avg_interval_time = -np.mean(interval_time)
        rew = np.exp(avg_interval_time)
        return rew, {'avg_interval_time': avg_interval_time, 'rew': rew}

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

        # total_reward = rew + conflict_penalty

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

    def estimated_end_time(self, j):
        '''
        calculate the estimated end time of job j
        :param j:
        :return:
        '''
        CKs = []
        for m in self.machines:
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