import numpy as np
from Environments.workShop import JobShop

class observation(JobShop):
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

        avail = np.zeros((self.num_machines, self.num_jobs+1))
        ready_opera = self.get_ready_opera()
        if len(ready_opera) == 0:
            for j in range(self.num_machines):
                if self.machines[j].available:  # the machine is available
                    avail[j][0] = 1
        else:
            for j in range(self.num_machines):
                if self.machines[j].available:  # the machine is available
                    if self.machines[j].currentTime > self.env.now:  # the machine is busy
                        avail[j][0] = 1
                    else:
                        i = 1
                        for o in ready_opera:
                            if o is not None:
                                if self.machines[j].name in o.cMachines.keys():
                                    avail[j][i] = 1
                            i += 1
                        if sum(avail[j][:]) == 0:
                            avail[j][0] = 1

        return np.array(avail), ready_opera

    def get_ready_opera(self):
        counter = 0
        ready_opera = []
        ready_o = [None] * self.num_jobs
        if len(self.jobs) != 0:
            finish_counter = 0
            for j in self.jobs:
                pre_opera = j.operation_list[0]
                for o in j.operation_list:
                    if not o.completed:
                        if o.idOpertion == 0:
                            ready_o[counter] = o
                        else:
                            if pre_opera.endTime <= self.env.now:
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