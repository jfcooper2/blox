from .placement import Placement
from .utils import *
import pandas as pd
import numpy as np
import cvxpy as cp

class PlacementGavel(Placement):
    """
    Implements Gavel placement policy
    """

    def __init__(self, args):
        """
        Use this to hold any extra state the scheduler wants to hold
        Note: not sure if necessary for Gavel
        """
        # TODO: Make this not depend on GPUs in multiple places
        self.rounds_received = pd.DataFrame(index=["V100", "P100", "K80"])
        self.throughputs = pd.DataFrame(index=["V100", "P100", "K80"])
        self.rounds_scheduled = 0
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

        # TODO: Is this needed?
        if len(active_jobs) == 0 and len(job_order) == 0:
            return (jobs_to_terminate, jobs_to_launch)

        if gpu_df.shape[1] < len(job_order):
            curr_jobs = job_order[:gpu_df.shape[1]]
        else:
            curr_jobs = job_order
        curr_jobs = [elem[0] for elem in curr_jobs]

        # Get the GPU throuputs into a nice dataframe
        for job in active_jobs.keys():
            if job not in self.throughputs.columns:
                self.throughputs[job] = [0, 0, 0]
                for gpu in self.throughputs.index:
                    self.throughputs[job][gpu] = active_jobs[job]["gpu_tputs"][gpu]

        # Make sure rounds received exists for each curr job
        for job in active_jobs.keys():
            if job not in self.rounds_received.columns:
                self.rounds_received[job] = 3 * [0.01]

        # Get the jobs in rounds_received that are curr_jobs
        curr_throughputs = self.throughputs[curr_jobs]
        curr_rounds_received = self.rounds_received[curr_jobs].T

        # TODO: Make these smarter
        curr_scale_factors = np.ones((len(curr_jobs), 3))
        curr_priority_weights = np.ones(len(curr_jobs))

        # TODO: THIS IS X
        curr_priorities = self.get_allocation(curr_throughputs.T, curr_scale_factors, curr_priority_weights, gpu_df)
        # print(curr_priorities)

        # Make a list of tuples from priorities and loop over these
        job_names = curr_throughputs.columns
        gpu_names = curr_throughputs.index

        # TODO: Check this is right
        job_priorities = [(curr_priorities[i][j] / curr_rounds_received.values[i][j], (job_name, gpu_name)) for i, job_name in enumerate(job_names) for j, gpu_name in enumerate(gpu_names)]
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
                if job["running_accel"] == gpu_preference:
                    # nothing to do here
                    placed_jobs.add(job_id)
                    
                    # bookkeeping
                    self.rounds_received[job_id][gpu_preference] += 1
                    continue
                else:
                    # need to terminate this job trying to launch on
                    # different accelerator
                    jobs_to_terminate.append(job_id)
                    job["is_running"] = False
                    delete_job_by_id(gpu_df, job_id)

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

                    # TODO: I think we don't do this? We may change our algorithm to not be greedy and this might be better
                    # no free GPUs
                    # find the GPU with same GPU preference in the reverse
                    # order of priority
                    """
                    for rev_idx in range(1, len(active_jobs) - idx):
                        potential_terminate_job_pair = job_order[-rev_idx]
                        if potential_terminate_job_pair[1] != gpu_preference:
                            # Job doesn't have the same preference
                            continue
                        else:
                            # the job has the same preference

                            # need to check if it is running
                            # and if it is running on the same as current
                            # preference
                            potential_terminate_job_info = active_jobs[
                                potential_terminate_job_pair[0]
                            ]
                            if (
                                potential_terminate_job_info["is_running"] == True
                            ) and (
                                potential_terminate_job_info["running_accel"]
                                == gpu_preference
                            ):
                                # only terminate in case the training is
                                # also happening on the same GPU as the
                                # preference
                                jobs_to_terminate.append(
                                    potential_terminate_job_pair[0]
                                )
                                potential_terminate_job_info["is_running"] = False
                                # freeing up GPUs
                                delete_job_by_id(
                                    gpu_df, potential_terminate_job_pair[0]
                                )
                                free_gpus = find_free_GPUs_by_type(
                                    gpu_df, gpu_preference
                                )

                                if place_consolidated:
                                    placement, found = self._consolidated_placement(
                                        job, free_gpus
                                    )
                                else:
                                    placement, found = self._scattered_placement(
                                        job, free_gpus
                                    )

                                if found:
                                    # we found the placement
                                    break

                                # terminate this job
                            else:
                                # job matching not found
                                continue
                    """
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

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
    # TODO: Priority_weights need to be computed externally. Per job, 1/priority
    # TODO: Scale_factors as well
    def get_allocation(self, throughputs, scale_factors, priority_weights, gpu_df):
        if throughputs is None:
            return None
        
        (m, n) = throughputs.shape
        #(job_ids, worker_types) = index

        #proportional_throughputs = self._proportional_policy.get_throughputs(
        #    throughputs, index, cluster_spec
        #)
        #priority_weights = np.multiply(
        #    priority_weights.reshape((m, 1)),
        #    1.0 / proportional_throughputs.reshape((m, 1)),
        #)

        x = cp.Variable(throughputs.shape)
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        objective = cp.Maximize(
            cp.sum(
                cp.multiply(
                    np.multiply(
                        throughputs * priority_weights.reshape((m, 1)),
                        scale_factors, 
                    ),
                    x,
                )
            )
            # cp.min(
            #     cp.sum(
            #         cp.multiply(
            #             np.multiply(
            #                 throughputs * priority_weights.reshape((m, 1)),
            #                 scale_factors, 
            #             ),
            #             x,
            #         ),
            #         axis=1,
            #     )
            # )
        )
        # Make sure that the allocation can fit in the cluster.
        constraints = [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors, x), axis=0) <= gpu_df.shape[0] / 3,
            cp.sum(x, axis=1) <= 1,
        ]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve()
        #result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print("WARNING: Allocation returned by policy not optimal!")

        alloc = x.value.clip(min=0.0).clip(max=1.0)

        self.rounds_scheduled += 1
        # if self.rounds_scheduled % 5 == 0:
        #     print(throughputs, scale_factors, priority_weights, gpu_df, alloc, sep='\n')
        #     # exit(1)
        
        return alloc

if __name__ == "main":
    pg = PlacementGavel()
    print(pg.get_allocation())
