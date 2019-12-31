#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import numpy as np
import ray
# from ray.cluster_utils import Cluster
from ray.tune.config_parser import make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    rllib train --run DQN --env CartPole-v0
Grid search example via RLlib CLI:
    rllib train -f tuned_examples/cartpole-grid-search-example.yaml
Grid search example via executable:
    ./train.py -f tuned_examples/cartpole-grid-search-example.yaml
Note that -f overrides all other trial-specific command-line options.
"""


def on_episode_start(info):
    print(info.keys()) # -> "env", 'episode'
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))
    episode.user_data["is_success"] = []


def on_episode_step(info):
    episode = info["episode"]
    is_success = episode.last_observation_for()[-1]
    is_success_raw = episode.last_raw_obs_for()[-1]
    assert is_success == is_success_raw
    episode.user_data["is_success"].append(is_success)
    # print("step: ")
    # print(is_success)


def on_episode_end(info):
    episode = info["episode"]
    successes = np.mean(episode.user_data["is_success"])
    episode.custom_metrics["success_mean"] = successes
    num = len(episode.user_data["is_success"])
    print("successes: ", successes)
    print("num of episode: ", num)
    if successes > 0:
        episode.custom_metrics["success_rate"] = 1.
    else:
        episode.custom_metrics["success_rate"] = 0.
    print("episode {} ended with length {} and success {}".format(episode.episode_id, episode.length, successes))
    

def on_sample_end(info):
    print("returned sample batch of size {}".format(info["samples"].count))


def on_train_result(info):
    print("trainer.train() result: {} -> {} episodes".format(
        info["trainer"], info["result"]["episodes_this_iter"]))
    # you can mutate the result dict to add new fields to return
    info["result"]["callback_ok"] = True


def on_postprocess_traj(info):
    episode = info["episode"]
    batch = info["post_batch"]
    print("postprocessed {} steps".format(batch.count))
    if "num_batches" not in episode.custom_metrics:
        episode.custom_metrics["num_batches"] = 0
    episode.custom_metrics["num_batches"] += 1



def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
        "of starting a new one.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="Emulate multiple cluster nodes for debugging.")
    parser.add_argument(
        "--ray-redis-max-memory",
        default=None,
        type=int,
        help="--redis-max-memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-memory",
        default=None,
        type=int,
        help="--memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="--object-store-memory to use if starting a new cluster.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_RESULTS_DIR,
        type=str,
        help="Local dir to save training results to. Defaults to '{}'.".format(
            DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--upload-dir",
        default="",
        type=str,
        help="Optional URI to sync training results to (e.g. s3://bucket).")
    parser.add_argument(
        "-v", action="store_true", help="Whether to use INFO level logging.")
    parser.add_argument(
        "-vv", action="store_true", help="Whether to use DEBUG level logging.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Whether to attempt to enable TF eager execution.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Whether to attempt to enable tracing for eager mode.")
    parser.add_argument(
        "--env", default="HuskyPickAndPlace-v1", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--queue-trials",
        action="store_true",
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    return parser


def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)
    else:
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": "PPO", # args.run,
                "env": "HuskyPickAndPlace-v1",
                "checkpoint_freq": 20, # args.checkpoint_freq,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial and
                    resources_to_json(args.resources_per_trial)),
                "stop": args.stop,
                # "config": {dict(args.config, env=args.env)},
                    "config": {"num_workers":5,
                    "callbacks": {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step,
                    "on_episode_end": on_episode_end,
                    # "on_sample_end": on_sample_end,
                    # "on_train_result": on_train_result,
                    # "on_postprocess_traj": on_postprocess_traj,
                },
            },
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    # verbose = 1
    # for exp in experiments.values():
    #     if not exp.get("run"):
    #         parser.error("the following arguments are required: --run")
    #     if not exp.get("env") and not exp.get("config", {}).get("env"):
    #         parser.error("the following arguments are required: --env")
    #     if args.eager:
    #         exp["config"]["eager"] = True
    #     if args.v:
    #         exp["config"]["log_level"] = "INFO"
    #         verbose = 2
    #     if args.vv:
    #         exp["config"]["log_level"] = "DEBUG"
    #         verbose = 3
    #     if args.trace:
    #         if not exp["config"].get("eager"):
    #             raise ValueError("Must enable --eager to enable tracing.")
    #         exp["config"]["eager_tracing"] = True

    # if args.ray_num_nodes:
    #     cluster = Cluster()
    #     for _ in range(args.ray_num_nodes):
    #         cluster.add_node(
    #             num_cpus=args.ray_num_cpus or 1,
    #             num_gpus=args.ray_num_gpus or 0,
    #             object_store_memory=args.ray_object_store_memory,
    #             memory=args.ray_memory,
    #             redis_max_memory=args.ray_redis_max_memory)
    #     ray.init(address=cluster.address)
    # else:
    ray.init(
        address=args.ray_address,
        object_store_memory=args.ray_object_store_memory,
        memory=args.ray_memory,
        redis_max_memory=args.ray_redis_max_memory,
        num_cpus=args.ray_num_cpus,
        num_gpus=args.ray_num_gpus)
    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume,
        verbose=2,)
        # concurrent=True)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num-iters", type=int, default=2000)
#     args = parser.parse_args()

#     ray.init()
#     trials = tune.run(
#         "PPO",
#         stop={
#             "training_iteration": args.num_iters,
#         },
#         config={
#             "env": "HuskyPickAndPlace-v1",
#             "num_gpus": 0,
#             "num_workers": 30,
#             # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#             "eager": False,
#             # "checkpoint": 20,
#             "callbacks": {
#                 "on_episode_start": on_episode_start,
#                 "on_episode_step": on_episode_step,
#                 "on_episode_end": on_episode_end,
#                 # "on_sample_end": on_sample_end,
#                 # "on_train_result": on_train_result,
#                 # "on_postprocess_traj": on_postprocess_traj,
#             },
#         },
#         return_trials=True)

#     # verify custom metrics for integration tests
#     custom_metrics = trials[0].last_result["custom_metrics"]
#     print(custom_metrics)
#     # assert "pole_angle_mean" in custom_metrics
#     # assert "pole_angle_min" in custom_metrics
#     # assert "pole_angle_max" in custom_metrics
#     # assert "num_batches_mean" in custom_metrics
#     # assert "callback_ok" in trials[0].last_result
