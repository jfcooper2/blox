from .placement import Placement
from .utils import *
import pandas as pd

class PlacementGavel(Placement):
    """
    Implements Gavel placement policy
    """

    def __init__(self, args):
        """
        Use this to hold any extra state the scheduler wants to hold
        Note: not sure if necessary for Gavel
        """
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

        raise NotImplementedError() 
        # TODO this implementation of gavel placement appears to be out of date
        # we get errors right off the bat, suggesting the infrastructure of blox
        # may have changed

        job_order = new_job_schedule["job_order"]
        scheduler = new_job_schedule.get("scheduler")
        jobs_to_terminate = list()
        jobs_to_launch = dict()
        launched_job_ids = list()

        # go over jobs in job order
        for idx, job_priority_sorted in enumerate(job_order):
            job_id, gpu_preference = list(job_priority_sorted.keys())[0]
            job = active_jobs[job_id]
            found = False
            if job["is_running"] == True:
                if job["running_accel"] == gpu_preference:
                    # nothing to do here
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
                    # no free GPUs
                    # find the GPU with same GPU preference in the reverse
                    # order of priority
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
            if found:
                launched_job_ids.append(job_id)
                jobs_to_launch[job_id] = placement
                active_jobs[jid]["running_accel"] = gpu_preference
                mark_gpu_in_use(gpu_df, placement, job_id)
            else:
                break

        return (jobs_to_terminate, jobs_to_launch)
