
class Machine():
    def __init__(self, id_mahicne, factory_id):
        self.idMachine = id_mahicne  # serial number of the machine
        self.located_factory = factory_id
        self.name = 'M' + str(self.idMachine)   # name of the machine
        self.currentTime = 0   # The completion time of the last operation processed on the machine. If the current time is less than or equal to  the now time of the enviornment, it means that the machine is idle.
        self.assignedOpera = []   # A list of operations that have been assigned to the machine
        self.state = 'idle' # the state of the machine, if 'idle', oo operations are currently being processed on the machine
        self.busyTime = 0   # sum of duration of all operations assigned in the machine
        self.available = True   # Whether the machine is available. When it is False, it means that the machine is faulty

