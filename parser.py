import adios as ad

print (ad.__version__)
method = "BP"
init = "verbose=3;"
num_steps = 1

ad.read_init(method, parameters=init)

f = ad.file("data/tau-metrics-updated/tau-metrics.bp", method, is_stream=True, timeout_sec = 10.0)

i = 0
while True:
	print(">>> step:", i)
	vname = 'event_timestamps'
	if vname in f.vars:
		event   = f.var[vname].read(nsteps=num_steps)
		print(event.shape)

	vname = 'counter_values'
	if vname in f.vars:
		event   = f.var[vname].read(nsteps=num_steps)
		print(event.shape)
	
	vname = 'comm_timestamps'
	if vname in f.vars:
		event   = f.var[vname].read(nsteps=num_steps)
		print(event.shape)

	print(">>> advance ... ")
	if (f.advance() < 0):
		break

	i += 1

f.close()

print(">>> Done.")
