"""Isaac Lab random agent that also reports how many env steps ran and how long they took.

Usage (from sim/cloud):
    ./push.sh examples/step_counter
    ./run_headless.sh step_counter/run_counted.py --task Isaac-Cartpole-Camera-Direct \
        --num_envs 4 --max_steps 300 --viz none physics=isaacsim_physx renderer=isaacsim_rtx \
        env.write_image_to_file=True
"""

import sys
import time

import warp as wp

wp.config.enable_backward = False

from isaaclab.envs import DirectRLEnv  # noqa: E402
from isaaclab_rl.entrypoints.simple_agents import run  # noqa: E402

steps = 0
first_step_at = None
original_step = DirectRLEnv.step


def counted_step(self, action):
    global steps, first_step_at
    if first_step_at is None:
        first_step_at = time.time()
    result = original_step(self, action)
    steps += 1
    if steps % 100 == 0:
        # printed from inside the loop because Kit's shutdown ends the process before run() returns
        print(f"PROOF_STEPS={steps} STEP_SECONDS={time.time() - first_step_at:.1f}", flush=True)
    return result


DirectRLEnv.step = counted_step
run(sys.argv[1:], policy="random")
