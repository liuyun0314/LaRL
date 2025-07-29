
class Operation():
    def __init__(self, id_operation, jobID, global_jobID, taskID, cMachineNames, arrival_time):
        self.taskID = taskID  # id of the task to which the operation belongs
        self.jobID = jobID  # id of the job to which the operation belongs
        self.global_jobID = global_jobID  # global id of the job to which the operation belongs
        self.idOpertion = id_operation   # serial number of the operation
        self.OperationName = 'O' + str(self.idOpertion)    # name of the operation
        self.AT = arrival_time  # arrival time of the operation
        self.weight = 1  # weight of the operation
        self.startTime = 0  # start time of the operation
        self.duration = 0  # duration of the operation. It is 0 when initialized and is 0 until the operation is assigned.
        self.endTime = 0  # end time of the operation
        self.cMachines = cMachineNames   # dictionary of candidate machines{'M1':89,,,,}, where M1 is the name of the machine and 89 is the processing time of the operation on the machine
        self.assignedMachine = ""    # name of the machine assigned to the operation
        self.completed = False  # True if the operation is completed, False otherwise
        self.assigned = False  # True if the operation is assigned to a machine, False otherwise

    def getEndTime(self):  # function to calculate the end time of the operation
        self.endTime = self.startTime + self.duration
        return self.endTime

