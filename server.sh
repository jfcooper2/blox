#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator.py --sim-type synthetic --jobs-per-hour 1 --start-job-track 0 --end-job-track 400 --exp-prefix test --placement Default --scheduler Las
#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator.py --sim-type synthetic --jobs-per-hour 1 --start-job-track 0 --end-job-track 400 --exp-prefix test --placement Gavel --scheduler Las
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator.py --sim-type synthetic --jobs-per-hour 1 --start-job-track 0 --end-job-track 400 --exp-prefix test --placement Best --scheduler Las
#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator.py --sim-type synthetic --jobs-per-hour 1 --start-job-track 0 --end-job-track 400 --exp-prefix test --placement Greedy --scheduler Las
#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator.py --cluster-job-log ./cluster_job_log --sim-type trace-synthetic --jobs-per-hour 1 --start-job-track 0 --end-job-track 400 --exp-prefix test
