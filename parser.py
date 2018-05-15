import adios as ad
import numpy as np

print (ad.__version__)
method = "BP"
init = "verbose=3;"
num_steps = 1

ad.read_init(method, parameters=init)

f = ad.file("data/tau-metrics-updated/tau-metrics.bp", method, is_stream=True, timeout_sec=10.0)

i = 0
while True:
	print(">>> step:", i)

	vname = 'event_timestamps'
	if vname in f.vars:
		var = f.var[vname]
		num_steps = var.nsteps
		event  = var.read(nsteps=num_steps)
		print(event.shape)
		data_event = np.zeros((event.shape[0], 12), dtype=object) + np.nan
		data_event[:, 0:5] = event[:, 0:5]
		data_event[:, 11] = event[:, 5]
		print(data_event[0])

	vname = 'counter_values'
	if vname in f.vars:
		var = f.var[vname]
		num_steps = var.nsteps
		counter  = var.read(nsteps=num_steps)
		print(counter.shape)
		data_counter = np.zeros((counter.shape[0], 12), dtype=object) + np.nan
		data_counter[:, 0:3] = counter[:, 0:3]
		data_counter[:, 5:7] = counter[:, 3:5]
		data_counter[:, 11] = counter[:, 5]
		print(data_counter[0])
	
	vname = 'comm_timestamps'
	if vname in f.vars:
		var = f.var[vname]
		num_steps = var.nsteps
		comm  = var.read(nsteps=num_steps)
		print(comm.shape)
		data_comm = np.zeros((comm.shape[0], 12), dtype=object) + np.nan
		data_comm[:, 0:4] = comm[:, 0:4]
		data_comm[:, 8:11] = comm[:, 4:7]
		data_comm[:, 11] = comm[:, 7]
		print(data_comm[0])

	print(">>> advance ... ")
	if (f.advance() < 0):
		break

	i += 1

f.close()

print(">>> Done.")
