
"""
Dynamic Streaming Parser
Author: Dingwen Tao (ustc.dingwentao@gmail.com)
Create: April, 2018
Modified: June, 2018
"""

# prog_names is the indexed set of program names in the attributes
# comm_ranks is the MPI rank
# threads is the thread ID (rank)
# event_types is the indexed set of event types in the attributes
# func_names is the indexed set of timer names in the attributes
# counters is the indexed set of counter names in the attributes
# event_types_comm is the indexed  set of event types related to communication in the attributes
# tag is the MPI tag
# partner is the other side of a point-to-point communication
# num_bytes is the amount of data sent

from collections import deque as dq
from collections import Counter as ct
import pickle
import itertools
import adios as ad
import numpy as np
import scipy.io as sio
import configparser


def Parser(configFile):
	method = "BP"
	init = "verbose=3;"

	# read parameters from configuration file
	config = configparser.ConfigParser()
	config.read(configFile)
	in_bp_file = config['Parser']['InputBPFile'] # input bp file path
	prov_db_path = config['Parser']['ProvDBPath'] # provenance output database path
	queue_size = int(config['Parser']['QueueSize']) # provenance data size
	int_func_num = int(config['Parser']['InterestFuncNum']) # interested function size

	# initialize adios streaming mode
	ad.read_init(method, parameters=init)
	fin = ad.file(in_bp_file, method, is_stream=True, timeout_sec=10.0)
	fout = open(prov_db_path, "wb")

	# read attributes
	db = dq(maxlen=queue_size)
	name = np.array(['prog_names', 'comm_ranks', 'threads', 'event_types', 'func_names', 'counters', 'counter_value', 'event_types_comm', 'tag', 'partner', 'num_bytes', 'timestamp']).reshape(1, 12)
	attr = fin.attr
	nattrs = fin.nattrs
	attr_name = list(fin.attr)
	attr_value = np.empty(nattrs, dtype=object)
	num_func = 0
	func_name = []
	for i in range(0, len(attr_name)):
		attr_value[i] = attr[attr_name[i]].value
		# count function number and names
		if attr_name[i].startswith('timer'):
			num_func = num_func + 1
			func_name.append(attr_value[i])
	attr_name = np.array(attr_name)
	func_name = np.array(func_name)

	i = 0
	while True:
		print(">>> step:", i)

		vname = "event_timestamps"
		if vname in fin.vars:
			var = fin.var[vname]
			num_steps = var.nsteps
			event  = var.read(nsteps=num_steps)
			data_event = np.zeros((event.shape[0], 12), dtype=object) + np.nan
			data_event[:, 0:5] = event[:, 0:5]
			data_event[:, 11] = event[:, 5]
			data_step = data_event
			# count most common functions
			int_func = ct(data_event[:, 4]).most_common(int_func_num) # e.g., [(16, 14002), (15, 14000), (13, 6000),...]
			# if i == 0:
			# 	data_step = data_event
			# else:
			# 	data_step = np.concatenate((data_step, data_event), axis=0)

		vname = "counter_values"
		if vname in fin.vars:
			var = fin.var[vname]
			num_steps = var.nsteps
			counter  = var.read(nsteps=num_steps)
			data_counter = np.zeros((counter.shape[0], 12), dtype=object) + np.nan
			data_counter[:, 0:3] = counter[:, 0:3]
			data_counter[:, 5:7] = counter[:, 3:5]
			data_counter[:, 11] = counter[:, 5]
			data_step = np.concatenate((data_step, data_counter), axis=0)

		vname = "comm_timestamps"
		if vname in fin.vars:
			var = fin.var[vname]
			num_steps = var.nsteps
			comm  = var.read(nsteps=num_steps)
			data_comm = np.zeros((comm.shape[0], 12), dtype=object) + np.nan
			data_comm[:, 0:4] = comm[:, 0:4]
			data_comm[:, 8:11] = comm[:, 4:7]
			data_comm[:, 11] = comm[:, 7]
			data_step = np.concatenate((data_step, data_comm), axis=0)

		print("Size of current timestep =", data_step.shape)

		# sort data in this step by timestamp
		data_step = data_step[data_step[:, 11].argsort()]

		# lauch anomaly detection
		flag = False

		# dynamic interest list
		if len(int_func) < 3:
			print ("Most interested function:\n", func_name[int_func[0][0]])
		else:
			print ("Most three interested functions:\n", func_name[int_func[0][0]], "\n", func_name[int_func[1][0]], "\n", func_name[int_func[2][0]])

		# ...

		# simulate an anomaly
		if i == 8:
			flag = True
		# ....

		# add or dump queue
		if flag:
			# dump queue to file	
			db.appendleft(attr_value)
			db.appendleft(attr_name)
			db.appendleft(nattrs)
			print(">>> Identified anomalies and dump data to binary.")
			print(">>> Serialization ...")
			pickle.dump(db, fout)
			# db[0]: the number of attributes
			# db[1]: the names of attributes
			# db[2]: the values of attributes
			# from db[3]: the trace data
		else:
			# add data to queue
			db.extend(data_step)


		print(">>> Advance to next step ... ")
		if (fin.advance() < 0):
			break

		i += 1

	fin.close()
	fout.close()

	print(">>> Complete passing data.")
	print(">>> Test of deserialization.")
	print(">>> Load data ...")
	fin = open(prov_db_path, "rb")
	db2 = pickle.load(fin)
	print("**** Print info ****")
	print("Number of attributes =", db2[0])
	print("First 20 Names of attributes =", db2[1][0:20])
	print("First 20 Values of attributes =", db2[2][0:20])
	print("First 20 trace data =", np.array(list(itertools.islice(db2, 3, 20))))
	fin.close()
	
	# import json
	# file_path = "data.json"
	# data_step = data_step[data_step[:, 11].argsort()]
	# with open('data.json', 'w') as outfile:
	# 	json.dump(data_step.tolist(), outfile)
	# outfile.close()
