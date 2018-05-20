from collections import deque as dq
import pickle
import adios as ad
import numpy as np
import scipy.io as sio

method = "BP"
init = "verbose=3;"
queue_size = 400000

ad.read_init(method, parameters=init)
fin = ad.file("data/tau-metrics-updated/tau-metrics.bp", method, is_stream=True, timeout_sec=10.0)
fout = open("data.pkl", "wb")

db = dq(maxlen=queue_size)
name = np.array(['prog_names', 'comm_ranks', 'threads', 'event_types', 'func_names', 'counters', 'counter_value', 'event_types', 'tag', 'partner', 'num_bytes', 'timestamp']).reshape(1, 12)
# prog_names is the indexed set of program names in the attributes
# comm_ranks is the MPI rank
# threads is the thread ID (rank)
# event_types is the indexed set of event types in the attributes
# func_names is the indexed set of timer names in the attributes
# counters is the indexed set of counter names in the attributes
# tag is the MPI tag
# partner is the other side of a point-to-point communication
# num_bytes is the amount of data sent
attr = fin.attr
nattrs = fin.nattrs
attr_name = list(fin.attr)
attr_value = np.empty(nattrs, dtype=object)
for i in range(0, len(attr_name)):
	attr_value[i] = attr[attr_name[i]].value
attr_name = np.array(attr_name)

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

	#lauch anomaly detection
	flag = False
	# ....
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

print(">>> Complete passing data.")

print(">>> Test of deserialization.")
print(">>> Load data ...")
fin = open("data.pkl", "rb")
db2 = pickle.load(fin)
print("**** Print info ****")
import itertools
print("Number of attributes =", db2[0])
print("First 20 Names of attributes =", db2[1][0:20])
print("First 20 Values of attributes =", db2[2][0:20])
print("First 20 trace data =", np.array(list(itertools.islice(db2, 3, 20))))