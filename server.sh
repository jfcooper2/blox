#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator.py --sim-type synthetic --jobs-per-hour 1 --start-job-track 0 --end-job-track 100 --exp-prefix test --placement Default
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator.py --sim-type synthetic --jobs-per-hour 1 --start-job-track 0 --end-job-track 100 --exp-prefix test --placement Gavel
#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator.py --cluster-job-log ./cluster_job_log --sim-type trace-synthetic --jobs-per-hour 1 --start-job-track 0 --end-job-track 10 --exp-prefix test
