import numpy as np

class Factory:
    def __init__(self, factory_id, mac_list):
        self.factory_id = factory_id
        self.num_machines = len(mac_list)
        self.machines = mac_list    # list of machines in the factory
        self.assigned_jobs = []    # list of queues for jobs waiting to be processed by machines in the factory
        self.idle_ratios = 0.0
        self.avg_UR = 0.0
        self.finished_jobs = []

    def get_UR(self, now):
        URs = []
        for m in self.machines:
            if len(m.assignedOpera) > 0:
                busy_time = 0
                for op in m.assignedOpera:
                    busy_time = busy_time + op.duration
                UR = busy_time/now
                URs.append(UR)
            else:
                return 0.0
        return np.mean(URs)

    def get_idle_ratio(self, now):
        idel = 0
        for m in self.machines:
            if m.currentTime <= now:
                idel = idel+1
        idel_ratio = idel/len(self.machines)
        return round(idel_ratio, 2)

    def estimated_average_delay(self):
        if len(self.assigned_jobs) > 0:
            CKs = []
            dely_num = 0
            for m in self.machines:
                CKs.append(m.currentTime)
            T_cure = np.mean(CKs)
            for j in self.assigned_jobs:
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
                dely = Cmax - j.DT
                if dely > 0:
                    dely_num = dely_num + dely
            delay_ratio = dely_num/len(self.assigned_jobs)
            return round(delay_ratio, 2)
        else:
            return 0.0

    def available_mac_time(self):
        mac_times = []
        for m in self.machines:
            m_inf0 = {}
            m_inf0[m.name] = m.currentTime
            mac_times.append(m_inf0)
        return mac_times






