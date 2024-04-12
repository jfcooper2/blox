import pandas as pd
import copy
from typing import Tuple, List

from .placement import Placement
from .utils import *


class PlacementDefault(Placement):
    def __init__(self, args):
        pass

    @staticmethod
    def copy_arguments(function):
        def function_wrapper(
            self, job_state, cluster_state, new_job_schedule, **kwargs
        ):
            return function(
                self,
                job_state.active_jobs,
                copy.deepcopy(new_job_schedule),
                copy.deepcopy(cluster_state.server_map),
                copy.deepcopy(cluster_state.gpu_df),
                **copy.deepcopy(kwargs),
            )

        return function_wrapper

    @copy_arguments.__func__
    def place(
        self,
        active_jobs: dict,
        new_job_schedule: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        **kwargs,
    ) -> dict:
        """
        parses the sorted_jobs dictionary and calls relevant placement policy

        # CAUTION: This function makes in place changes to active jobs and
        # gpu_df

        """

        job_order = new_job_schedule["job_order"]
        scheduler = new_job_schedule.get("scheduler")
        jobs_to_terminate = list()
        jobs_to_launch = dict()
        launched_job_ids = list()

        # accel_sorted_by_pref - key: gpu_type, val: list of job ids sorted
        # by decreasing preference

        # no scheduler provided
        running_jobs = 0
        new_scheduled_jobs = 0
        jobs_to_schedule = 0
        for idx, job_id in enumerate(job_order):
            job_id, _ = job_id
            job = active_jobs[job_id]
            found = False
            if job["is_running"] == True:
                # move to lower priority jobs
                running_jobs += 1
                continue
            if job["is_running"] == False:
                # need to find placement only if job is not running
                place_consolidated = (
                    job.get("placement_preference") == "consolidated"
                )

                # first checking if there are free GPUs
                free_gpus = find_free_GPUs(gpu_df)
                if place_consolidated:
                    placement, found = self._consolidated_placement(job, free_gpus)
                else:
                    placement, found = self._scattered_placement(job, free_gpus)
                # next checking if there are lower priority jobs which have
                if not found:
                    # no free GPUs
                    # need to see if there are lower priority jobs which can be
                    # terminated and placement can be found then

                    for rev_idx in range(1, len(active_jobs) - idx):
                        potential_job_to_terminate = active_jobs[
                            job_order[-rev_idx][0]
                        ]
                        if potential_job_to_terminate["is_running"] == True:
                            # terminate this job
                            jobs_to_terminate.append(job_order[-rev_idx][0])
                            potential_job_to_terminate["is_running"] = False
                            # freeing up GPUs
                            delete_job_by_id(gpu_df, job_order[-rev_idx][0])
                            free_gpus = find_free_GPUs(gpu_df)
                            if place_consolidated:
                                placement, found = self._consolidated_placement(
                                    job, free_gpus
                                )
                            else:
                                placement, found = self._scattered_placement(
                                    job, free_gpus
                                )
                            if found:
                                # we found an assignment
                                # print(
                                # f"Placed {job_id} by determining to terminate{job_order[-rev_idx][0]}"
                                # )
                                break
            if found:
                new_scheduled_jobs += 1
                jobs_to_launch[job_id] = placement
                # update manual-pipeline-list for bert and gpt
                mark_gpu_in_use(gpu_df, placement, job_id)
            else:
                # print(f"New Jobs scheduled {new_scheduled_jobs}")
                # print(f"Jobs previously running {running_jobs}")
                # print(f"Jobs terminated {len(jobs_to_terminate)}")
                # print(f"Jobs in queue {len(job_order)-idx}")
                break
        return (jobs_to_terminate, jobs_to_launch)
