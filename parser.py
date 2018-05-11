import adios as ad
import getopt, sys
import os

filepath = "data/tau-metrics-updated/tau-metrics.bp"
f = ad.file(filepath, method_name='BP', is_stream=True, timeout_sec = 10.0)
f.printself()