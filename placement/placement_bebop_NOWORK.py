from .placement import JobPlacement
from .utils import *


class PlacementBebop(JobPlacement):
    def __init__(self, args):
        raise NotImplementedError("Bebop no work")

        """
        Placement Policy
        """
        from .throughput_predictor import ThrouhputPredictor, train

        self.tpredictor = ThrouhputPredictor(
            {1: ["v100", "v100", "v100", "v100"]},
            {1: ["Intel", "RAM", "Storage_bw"]},
            [
                "gan",
                "cifar10-res18",
                "imagenet-res50",
                "translation",
                "recommendation",
                "rl",
                "language_modeling",
            ],
            "/hdd1/bebop_data_2-3-4_20_perc/",  # represents percentage of training data used
        )
        avg_error, loss_val = self.tpredictor.validate()
        print(
            "In Placement Avg Error {}, In Placement Avg Loss {}".format(
                avg_error, loss_val
            )
        )

    def place(
        self,
        active_jobs: dict,
        new_job_schedule: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        **kwargs,
    ) -> dict:

        job_order = new_job_schedule["job_order"]
        scheduler = new_job_schedule.get("scheduler")
        jobs_to_terminate = list()
        job_to_launch = dict()
        launched_job_ids = list()
        # go over jobs in job order
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
                place_consolidated = job.get("placement_preference") == "consolidated"

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
                        potential_job_to_terminate = active_jobs[job_order[-rev_idx][0]]
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
                                print(
                                    f"Placed {job_id} by determining to terminate{job_order[-rev_idx][0]}"
                                )
                                break
            if found:
                new_scheduled_jobs += 1
                job_to_launch[job_id] = placement
                mark_gpu_in_use(gpu_df, placement, job_id)
            else:
                print(f"New Jobs scheduled {new_scheduled_jobs}")
                print(f"Jobs previously running {running_jobs}")
                print(f"Jobs terminated {len(jobs_to_terminate)}")
                print(f"Jobs in queue {len(job_order)-idx}")
                break
            return (jobs_to_terminate, job_to_launch)

