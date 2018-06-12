# Online Anomaly Detection for HPC Performance Data

This library provides a Python API to process [SOSflow](https://github.com/cdwdirect/sos_flow)(Scalable Observation System for Scientific Workflows) performance traces. It supports the following functionality:

- Streaming Event Parser: dynamically passes events of SOSflow with interest functions
- Streaming Anomaly Detector: detects anomalies in performance of functions based on limited memory incremental local outlier factor algorithm.

# Requirement
Our codebase requires Python 3.5 or higher and python and pip to be linked to Python 3.5 or higher.

# Installation
Run the following script: 'scripts/install-dependency.sh'

    bash scripts/install-dependency.sh

# Test
To run tests:

    make
    make test

# Example

[[1]] The following example code illustrates the basic usage of our online anomaly detection function.

First, configure the parameters in the [Analyzer] section of the configuration file (e.g., chimbuko.cfg). 

Then, call the only anomaly detection API by:

	from MiLOF import MILOF
	MILOF("chimbuko.cfg")
	
It will generate local outlier factor for each incoming data point.

	
[[2]] The following example code illustrates the basic usage of our dynamic event parser function.

First, configure the parameters in the [Parser] section of the configuration file (e.g., chimbuko.cfg). 

Then, call the dynamic event parser API by:

	from strmParser import Parser
	Parser("chimbuko.cfg")
	
It will output some information step by step as follows.

    >>> step: 0
    Size of current timestep = (48, 12)
    Most three interested functions:
     b'MPI_Init()'
     b'.TAU application'
     b'HEAT_TRANSFER [{heat_transfer.F90} {22,1}-{140,25}]'
    >>> Advance to next step ...
    >>> step: 1
    Size of current timestep = (421, 12)
    Most three interested functions:
     b'MPI_Comm_split()'
     b'pthread_create'
     b'Step[0]'
    >>> Advance to next step ...
    >>> step: 2
    Size of current timestep = (241, 12)
    Most three interested functions:
     b'Step[1]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Advance to next step ...
    >>> step: 3
    Size of current timestep = (250, 12)
    Most three interested functions:
     b'Step[2]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Advance to next step ...
    >>> step: 4
    Size of current timestep = (250, 12)
    Most three interested functions:
     b'Step[3]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Advance to next step ...
    >>> step: 5
    Size of current timestep = (258, 12)
    Most three interested functions:
     b'Step[4]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Advance to next step ...
    >>> step: 6
    Size of current timestep = (262, 12)
    Most three interested functions:
     b'Step[5]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Advance to next step ...
    >>> step: 7
    Size of current timestep = (272, 12)
    Most three interested functions:
     b'Step[6]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Advance to next step ...
    >>> step: 8
    Size of current timestep = (262, 12)
    Most three interested functions:
     b'Step[7]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Identified anomalies and dump data to binary.
    >>> Serialization ...
    >>> Advance to next step ...
    >>> step: 9
    Size of current timestep = (272, 12)
    Most three interested functions:
     b'Step[8]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Advance to next step ...
    >>> step: 10
    Size of current timestep = (382, 12)
    Most three interested functions:
     b'Step[9]'
     b'adios_open'
     b'MPI_Comm_dup()'
    >>> Advance to next step ...
    >>> step: 11
    Size of current timestep = (80, 12)
    Most three interested functions:
     b'MPI_Barrier()'
     b'HEAT_IO::IO_FINALIZE [{io_adios.F90} {27,1}-{31,26}]'
     b'adios_finalize'
    >>> Advance to next step ...
    >>> step: 12
    Size of current timestep = (18, 12)
    Most three interested functions:
     b'MPI_Finalize()'
     b'.TAU application'
     b'HEAT_TRANSFER [{heat_transfer.F90} {22,1}-{140,25}]'
    >>> Advance to next step ...
    >>> Complete passing data.
    >>> Test of deserialization.
    >>> Load data ...
    **** Print info ****
    Number of attributes = 518
    First 20 Names of attributes = ['program_name 0' 'MetaData:0:0:0:CPU Cores' 'MetaData:0:0:0:CPU MHz'
     'MetaData:0:0:0:CPU Type' 'MetaData:0:0:0:CPU Vendor'
     'MetaData:0:0:0:CWD' 'MetaData:0:0:0:Cache Size'
     'MetaData:0:0:0:Command Line' 'MetaData:0:0:0:Executable'
     'MetaData:0:0:0:Hostname' 'MetaData:0:0:0:Local Time'
     'MetaData:0:0:0:Memory Size' 'MetaData:0:0:0:Node Name'
     'MetaData:0:0:0:OS Machine' 'MetaData:0:0:0:OS Name'
     'MetaData:0:0:0:OS Release' 'MetaData:0:0:0:OS Version'
     'MetaData:0:0:0:Starting Timestamp' 'MetaData:0:0:0:TAU Architecture'
     'MetaData:0:0:0:TAU Config']
    First 20 Values of attributes = [b'/home/khuck/src/Example-Heat_Transfer/stage_write/stage_write' b'4'
     b'2667.000' b'Intel(R) Xeon(R) CPU X5355 @ 2.66GHz' b'GenuineIntel'
     b'/home/khuck/src/Example-Heat_Transfer/test_sos' b'4096 KB'
     b'../stage_write/stage_write heat.bp staged.bp FLEXPATH'
     b'/home/khuck/src/Example-Heat_Transfer/stage_write/stage_write' b'ktau'
     b'2018-06-11T11:51:52-07:00' b'8172400 kB' b'ktau' b'x86_64' b'Linux'
     b'4.4.0-127-generic' b'#153-Ubuntu SMP Sat May 19 10:58:46 UTC 2018'
     b'1528743112852053' b'default'
     b' -iowrapper -pdt=/home/khuck/install/pdtoolkit-3.25 -papi=/usr/local/papi/5.5.0 -sos=/home/khuck/install/sos_flow -mpi -adios=/home/khuck/src/ADIOS/ADIOS-gcc']
    First 20 trace data = [[0 0 0 nan nan 3 0 nan nan nan nan 0]
     [0 0 0 nan nan 2 15568 nan nan nan nan 0]
     [0 0 0 nan nan 1 15568 nan nan nan nan 0]
     [1 0 0 nan nan 0 91802 nan nan nan nan 0]
     [0 0 0 nan nan 0 91776 nan nan nan nan 0]
     [1 0 0 nan nan 1 16332 nan nan nan nan 0]
     [1 0 0 nan nan 3 0 nan nan nan nan 0]
     [1 0 0 nan nan 2 16332 nan nan nan nan 0]
     [1 1 0 nan nan 0 91804 nan nan nan nan 1000000]
     [1 1 0 nan nan 1 16288 nan nan nan nan 1000000]
     [1 1 0 nan nan 2 16288 nan nan nan nan 1000000]
     [1 1 0 nan nan 3 0 nan nan nan nan 1000000]
     [0 1 0 nan nan 3 0 nan nan nan nan 1000000]
     [0 1 0 nan nan 0 91776 nan nan nan nan 1000000]
     [0 1 0 nan nan 1 15620 nan nan nan nan 1000000]
     [0 1 0 nan nan 2 15620 nan nan nan nan 1000000]
     [1 3 0 nan nan 1 16332 nan nan nan nan 2000000]]
    Streaming event parser test passed.
