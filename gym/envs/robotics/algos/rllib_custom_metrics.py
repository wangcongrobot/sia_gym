"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a custom metric.

https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics

https://github.com/ray-project/ray/blob/master/rllib/evaluation/episode.py

"""

import argparse
import numpy as np

import ray
from ray import tune
'''
    # Callbacks that will be run during various phases of training. These all
    # take a single "info" dict as an argument. For episode callbacks, custom
    # metrics can be attached to the episode by updating the episode object's
    # custom metrics dict (see examples/custom_metrics_and_callbacks.py). You
    # may also mutate the passed in batch data in your callback.
    "callbacks": {
        "on_episode_start": None,     # arg: {"env": .., "episode": ...}
        "on_episode_step": None,      # arg: {"env": .., "episode": ...}
        "on_episode_end": None,       # arg: {"env": .., "episode": ...}
        "on_sample_end": None,        # arg: {"samples": .., "worker": ...}
        "on_train_result": None,      # arg: {"trainer": ..., "result": ...}
        "on_postprocess_traj": None,  # arg: {
                                      #   "agent_id": ..., "episode": ...,
                                      #   "pre_batch": (before processing),
                                      #   "post_batch": (after processing),
                                      #   "all_pre_batches": (other agent ids),
                                      # }
    },
'''

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


def on_episode_end(info):
    episode = info["episode"]
    successes = np.sum(episode.user_data["is_success"])
    if successes > 0.1:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=2000)
    args = parser.parse_args()

    ray.init()
    trials = tune.run(
        "PPO",
        stop={
            "training_iteration": args.num_iters,
        },
        config={
            "env": "HuskyPickAndPlace-v1",
            "num_gpus": 0,
            "num_workers": 30,
            # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "eager": False,
            # "checkpoint": 20,
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end,
                # "on_sample_end": on_sample_end,
                # "on_train_result": on_train_result,
                # "on_postprocess_traj": on_postprocess_traj,
            },
        },
        return_trials=True)

    # verify custom metrics for integration tests
    custom_metrics = trials[0].last_result["custom_metrics"]
    print(custom_metrics)
    # assert "pole_angle_mean" in custom_metrics
    # assert "pole_angle_min" in custom_metrics
    # assert "pole_angle_max" in custom_metrics
    # assert "num_batches_mean" in custom_metrics
    # assert "callback_ok" in trials[0].last_result
