import os 
import subprocess
import time
import pickle 

# map = "t1_triple" # t1_triple, t2_triple, t3, t4, shanghai_intl_circuit
map = "t2_triple"
# map = "t3"
# map = "t4"
# map = "shanghai_intl_circuit"

controller_process = subprocess.Popen("python3 automatic_control_GRAIC.py --sync -m {}".format(map), shell=True)
time.sleep(5)
# scenario_process = subprocess.Popen("python3 scenario.py -m {}".format(map), shell=True, stdout=None)
