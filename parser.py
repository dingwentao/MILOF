import adios as ad

method = "BP"
init = "verbose=3;"

ad.read_init(method, parameters=init)

f = ad.file("data/tau-metrics-updated/tau-metrics.bp", method, is_stream=True, timeout_sec = 10.0)

i = 0
while True:
    print(">>> step:", i)
    v = f.var['event_timestamps']
    print(v)

    val = v.read(nsteps=1)
    print(val.shape)

    print(">>> advance ... ")
    if (f.advance() < 0):
        break
    i += 1

f.close()

print(">>> Done.")
