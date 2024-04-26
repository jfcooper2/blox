from .placement import Placement
from .utils import *
import pandas as pd
import numpy as np
import cvxpy as cp

class PlacementGreedy(Placement):
    """
    Implements Gavel placement policy
    """

    def __init__(self, args):
        """
        Use this to hold any extra state the scheduler wants to hold
        Note: not sure if necessary for Gavel
        """
        # TODO: Make this not depend on GPUs in multiple places
        self.gpu_types = ["V100", "P100", "K80"]
        self.rounds_received = pd.DataFrame(index=self.gpu_types)
        self.throughputs = pd.DataFrame(index=self.gpu_types)
        self.rounds_scheduled = 0
        self.swap_count = 0
        pass # at the moment (2024/04/05), not sure if state is needed

    @Placement.copy_arguments
    def place(
        self,
        active_jobs: dict,
        new_job_schedule: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        **kwargs,
    ) -> dict:

        # TODO this implementation of gavel placement appears to be out of date
        # we get errors right off the bat, suggesting the infrastructure of blox
        # may have changed

        job_order = new_job_schedule["job_order"]
        scheduler = new_job_schedule.get("scheduler")
        jobs_to_terminate = list()
        jobs_to_launch = dict()
        launched_job_ids = list()
        num_gpu_types = len(self.gpu_types)

        # TODO: Is this needed?
        if len(active_jobs) == 0 and len(job_order) == 0:
            return (jobs_to_terminate, jobs_to_launch)

        if gpu_df.shape[0] < len(job_order):
            curr_jobs = job_order[:gpu_df.shape[0]]
        else:
            curr_jobs = job_order
        curr_jobs = [elem[0] for elem in curr_jobs]
        # This seems odd
        #curr_jobs = set(curr_jobs).union(gpu_df["JOB_IDS"].values[gpu_df["JOB_IDS"].values != None])

        # Get the GPU throuputs into a nice dataframe
        for job in active_jobs.keys():
            if job not in self.throughputs.columns:
                self.throughputs[job] = num_gpu_types * [0.0]
                for gpu in self.throughputs.index:
                    self.throughputs[job][gpu] = active_jobs[job]["gpu_tputs"][gpu]

        # Make sure rounds received exists for each curr job
        for job in active_jobs.keys():
            if job not in self.rounds_received.columns:
                # Avoid floating point errors
                self.rounds_received[job] = num_gpu_types * [0.01]

        # Get the jobs in rounds_received that are curr_jobs
        curr_throughputs = self.throughputs[curr_jobs]
        curr_rounds_received = self.rounds_received[curr_jobs].T

        # TODO: Make these smarter
        curr_scale_factors = np.ones((len(curr_jobs), num_gpu_types))
        curr_priority_weights = np.ones(len(curr_jobs))

        # Make a list of tuples from priorities and loop over these
        job_names = curr_throughputs.columns
        gpu_names = curr_throughputs.index

        # TODO: Check this is right
        print(curr_throughputs)
        job_priorities = [(curr_throughputs.values[j][i], (job_name, gpu_name)) for i, job_name in enumerate(job_names) for j, gpu_name in enumerate(gpu_names)]
        job_priorities.sort(key=lambda x: -x[0]) # Sort by priority

        if self.rounds_scheduled % 5 == 0:
            # print(f'rounds_scheduled {self.rounds_scheduled}')
            # print('rounds_received:')
            # print(curr_rounds_received)
            # print('priorities')
            # print(curr_priorities / curr_rounds_received.values)
            print('GPU-job tuples sorted by priority')
            print(job_priorities)
            # exit(1)

        # when a job is chosen to run, add it to this set
        placed_jobs = set()

        for job_id in np.unique(gpu_df["JOB_IDS"].values[gpu_df["JOB_IDS"].values != None]):
            if job_id not in curr_jobs:
                print("-------------- EVICTING ----------------")
                job = active_jobs[job_id]
                jobs_to_terminate.append(job_id)
                job["is_running"] = False
                delete_job_by_id(gpu_df, job_id)

        # go over jobs in job order
        for priority, (job_id, gpu_preference) in job_priorities:

            # check if job has already placed
            # if so, ignore it and move on to the next tuple
            if job_id in placed_jobs:
                print(f'We have already placed {job_id}')
                continue

            job = active_jobs[job_id]
            found = False
            if job["is_running"] == True:
                placed_jobs.add(job_id)
                self.rounds_received[job_id][gpu_preference] += 1
                continue

            if job_id in launched_job_ids:
                # already launched the same ID in this round
                continue
            if job["is_running"] == False:
                # need to find placement only if job is not running
                place_consolidated = (
                    job.get("placement_preference") == "consolidated"
                )

                free_gpus = find_free_GPUs_by_type(gpu_df, gpu_preference)
                if place_consolidated:
                    placement, found = self._consolidated_placement(job, free_gpus)
                else:
                    placement, found = self._scattered_placement(job, free_gpus)
                if not found:
                    continue

            if found:
                launched_job_ids.append(job_id)
                jobs_to_launch[job_id] = placement
                active_jobs[job_id]["running_accel"] = gpu_preference
                mark_gpu_in_use(gpu_df, placement, job_id)
                
                # bookkeeping
                placed_jobs.add(job_id)
                self.rounds_received[job_id][gpu_preference]+= 1
            else:
                break

        print('Placed jobs:')
        print(placed_jobs)

        return (jobs_to_terminate, jobs_to_launch)


if __name__ == "main":
    pg = PlacementGavel()
    print(pg.get_allocation())
