#=========================================================================================
import random
import numpy as np

class Job():
    """Represent job to-do in schedule"""

    def __init__(self, task_id, id_job, weight, arrival_time, operations_list, globale_id):
        self.idTask = task_id  # id of the task to which the job belongs
        self.idJob = id_job     # serial number of the job
        self.JobName = 'J'+ str(self.idJob)   # name of the job
        self.global_id = globale_id
        self.assigned_factory = None   # the factory assigned to the job
        self.weight = weight   # the weight of the job
        self.operation_list = operations_list   #  list of operations in a job
        self.endTime = 0    # the completion time of the last operation in the job. It is determined only after all operations belonging to the job have been completed
        self.assigned = False
        self.AT = arrival_time  # arrival time of a job
        self.RT = self.getReleaseTime()  # release time of a job, i.e. the earliest time at which a job can start processing. It is usually set to the start time of the first operation
        # self.span = self.getSpan()  # the process time of a job
        self.span = self.endTime - self.RT  # the process time of a job
        self.DDT = 1.5   # random.uniform(0.5, 1.5)  due date tightness, Zref{Real-Time Scheduling for Dynamic Partial-No-Wait Multiobjective Flexible Job Shop by Deep Reinforcement Learning}
        self.DT = self.getDueTime()       # due time of a job
        self.completed = False   #  a flag to indicate whether a job has been completed or not
        self.wait_time = self.RT - self.AT   # the waiting time of a job
        self.expected_time = self.estimatedCompletionTime()

    def __hash__(self):
        return hash(str(self))

    def getTupleStartAndDuration(self):
        return (self.RT, self.span)

    def getEndTime(self):
        endTime = 0
        if len(self.operation_list) != 0:
            for index in range(len(self.operation_list)):
                if self.operation_list[index].completed:
                    endTime = self.operation_list[index].endTime
        return endTime

    def getDueTime(self):
        dueTime = 0
        if len(self.operation_list) != 0:
            t_i_j = []
            for o in self.operation_list:
                cmachines = o.cMachines
                t_j = sum(cmachines.values()) / len(cmachines)
                t_i_j.append(t_j)
            dueTime = np.round(self.AT + self.DDT * sum(t_i_j))
        return dueTime

    def getReleaseTime(self):
        if len(self.operation_list) != 0:
            self.RT = self.operation_list[0].startTime
        self.wait_time = self.RT - self.AT
        return self.RT

    def estimatedCompletionTime(self):
        dueTime = 0
        if len(self.operation_list) != 0:
            t_i_j = []
            for o in self.operation_list:
                cmachines = o.cMachines
                t_j = sum(cmachines.values()) / len(cmachines)
                t_i_j.append(t_j)
            dueTime = np.round(self.AT + self.DDT * sum(t_i_j))
        return dueTime