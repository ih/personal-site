---
title: Sim2Real First Steps
date: 2024-09-01
description: You think that's air you're breathing now?
---
*Disclaimer: I used the [Waveshare Jetbot Kit](https://www.waveshare.com/product/ai/robots/mobile-robots/jetbot-ai-kit-acce.htm) with the [Jetson Nano computer](https://developer.nvidia.com/buy-jetson?product=jetson_nano&location=US), but the Jetson Nano has been [discontinued](https://jetbot.org/master/bill_of_materials.html). You can probably follow along with a Jetbot using [Jetson Orin Nano](https://jetbot.org/master/bill_of_materials_orin.html), but I haven’t tried it. Hopefully this post is still useful to those interested in setting up sim2real using Isaac Lab with whatever robot you choose.*

My only experience with robotics is from one undergraduate course I took 20 years ago so I’m very much a novice to the field. I was inspired by DeepMind’s [soccer playing robots](https://sites.google.com/view/op3-soccer?), which falls into the areas of deep reinforcement learning and training a system in simulation then transferring it to a real robot known as sim-to-real or sim2real. I didn’t find too much online about specific steps for getting a sim 2 real pipeline set up so I decided to write this blog post about my experience. 

After some research I decided to go with [Jetbot](https://jetbot.org/master/) and [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) for my first robot and simulation environment. Jetbot with the Jetson Nano is relatively inexpensive compared to other robots with a camera and motors and Isaac Sim already had a built-in model as well as a framework for setting up deep reinforcement learning environments called [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). The main focus of this project was to figure out how to train an agent in simulation and then put the learned agent into the physical robot. I picked driving forward as the task to learn to keep things simple. The workflow roughly follows the [official tutorials](https://isaac-sim.github.io/IsaacLab/source/tutorials/index.html), but I try to add details specifically for getting things to work with Jetbot that I didn’t find online. A word of warning, this isn’t a step by step tutorial or introduction to Jetbot or Isaac Lab. You’ll still probably need a familiarity that you’d get by going through the standard tutorials for each.

## Creating a Scene
The first step was to set up the 3D scene where learning will take place. This involved importing assets for Jetbot and a room, both of which are easily accessible from IsaacSim’s asset library. The basic way this works is you create “configuration” classes that act as wrappers around assets. In the case of Jetbot you have an [`ArticulationConfig`](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbot.py)
```
JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
    }
``` 
and for the room there is an [`AssetBaseCfg`](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L42).

```
AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd"))
```

These assets are further wrapped in an [interactive scene configuration](https://isaac-sim.github.io/IsaacLab/source/tutorials/02_scene/create_scene.html). And all these configurations are instantiated by the Isaac Lab framework.

[`JetBotSceneCfg`](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L41)
```
class JetbotSceneCfg(InteractiveSceneCfg):
    room_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd"))
    
    jetbot: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(-.6,0,0)))

    camera = CameraCfg(
        data_types=["rgb"],
        prim_path="{ENV_REGEX_NS}/Robot/chassis/rgb_camera/jetbot_camera",
        spawn=None,
        height=224,
        width=224,
        update_period=.1
    )

    goal_marker = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/marker", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd"), init_state=RigidObjectCfg.InitialStateCfg(pos=(.3,0,0)))
```

I also created a camera, which is used in the definition of observations, and a goal marker, which is used in the definition of reward.

## Defining the Reinforcement Learning Environment
The next step is to define the reinforcement learning environment. This involves defining functions to get observations, apply actions, distribute rewards, and set restart conditions and values. You can then choose a deep reinforcement learning library and algorithm (or define your own) to interface with Isaac Lab through these functions to solve your task. Isaac Lab lists two ways to create a reinforcement learning environment in the official tutorials. A [manager-based RL environment](https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_manager_rl_env.html) and a [direct workflow RL environment](https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_direct_rl_env.html). I found since I wanted to use camera input for observations it’d be easier to use the direct workflow. 

### Observations
I mainly followed and modified this camera based cartpole [example](https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/cartpole/cartpole_camera_env.py) for figuring out how to get observations from the camera. The main change was [slicing out the alpha channel](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L128) from the Jetbot camera.
```
    def _get_observations(self) -> dict:
        observations =  self.robot_camera.data.output["rgb"].clone()
        # get rid of the alpha channel
        observations = observations[:, :, :, :3]
        return {"policy": observations}
```

*Disclaimer: This set up successfully feeds camera data as input to the learning algorithm, but It’s worth noting the task of driving forward doesn’t actually need camera input (you just need to learn to apply the same velocities to both motors). It’s possible I’m missing something important in the setup if the image data is important to the task.*

### Actions and Rewards
Applying actions was pretty simple and just a matter of [setting the joint velocities](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L135) of the robot.
```
    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions)
```
The [reward](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L113) for the task was based on distance to a goal marker which was positioned directly in front of the robot at the start of an episode. Basically giving more reward the closer the robot is to the goal marker.

```
    def _get_rewards(self) -> torch.Tensor:
        robot_position = self.robot.data.root_pos_w
        goal_position = self.goal_marker.data.root_pos_w
        squared_diffs = (robot_position - goal_position) ** 2
        distance_to_goal = torch.sqrt(torch.sum(squared_diffs, dim=-1))
        rewards = torch.exp(1/(distance_to_goal))
        rewards -= 30

        if (self.common_step_counter % 10 == 0):
            print(f"Reward at step {self.common_step_counter} is {rewards} for distance {distance_to_goal}")
        return rewards
```

### Resetting
I ended up using only a [timeout](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L148) for the episode termination condition although initially planned to terminate if the robot was too far from the goal. 

```
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = .01
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        robot_position = self.robot.data.root_pos_w
        goal_position = self.goal_marker.data.root_pos_w
        squared_diffs = (robot_position - goal_position) ** 2
        distance_to_goal = torch.sqrt(torch.sum(squared_diffs, dim=-1))
        distance_within_epsilon = distance_to_goal < epsilon
        distance_over_limit = distance_to_goal > .31
        position_termination_condition = torch.logical_or(distance_within_epsilon, distance_over_limit)
        position_termination_condition.fill_(False) # just use the timeout
        return (position_termination_condition, time_out)
```

[Resetting the scene](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L151) for each episode was a pretty direct translation of examples from the tutorials and the camera cartpole code.

```
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        default_goal_root_state = self.goal_marker.data.default_root_state[env_ids]
        default_goal_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.goal_marker.write_root_pose_to_sim(default_goal_root_state[:, :7], env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
```

##  Learning the Task
I followed the cartpole camera example to set up learning with the [rl_games](https://github.com/Denys88/rl_games/tree/master) library. You need to register the task, which can be found in this [\_\_init\_\_.py](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/__init__.py) file and create the associated configuration files in the [jetbot folder](https://github.com/ih/JetbotSim2RealBasic/tree/main/jetbot/agents). I also slightly modified the [training script](https://github.com/ih/JetbotSim2RealBasic/blob/main/train.py#L57) for rl_games to import the Jetbot folder. I modified a few parameters like score_to_win, minibatch_size, and mini_epochs to get things to run, but didn’t experiment too much since the focus was on the sim2real aspect and not learning a complex task. That being said there were a few adjustments I had to make to learn the task.

One thing I found that made learning work for this particular task was to [limit the action space](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L103) to -1,1. I initially had the limits as -inf, inf, but the Jetbot would basically be stuck in place. After changing the action space and examining what was being learned I found out the network is essentially learning to output high values for left and right motors, which are then [clipped](https://github.com/Denys88/rl_games/blob/master/rl_games/common/a2c_common.py#L1169) to 1 producing the desired result.

The other thing was adding a constant [penalty](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot/jetbotenv.py#L119) to the reward function, which discouraged staying in place.

After you register the task and add the configuration files you can run the learning session from the console using Isaac Lab e.g.

`.\IsaacLab\isaaclab.bat -p .\JetbotSim2RealBasic\train.py –task Isaac-Jetbot-Direct-v0 –num_envs 1`

Here’s a video of the jetbot learning to move forward.

<iframe width="560" height="315" src="https://www.youtube.com/embed/pbsuXIrCXJI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

After training is complete a `.pth` file is created in the logging directory and this contains the weights for the model.

## Simulation to Reality

My original plan was to install rl_games onto the Jetbot and load the `.pth` file similar to the [play.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/standalone/workflows/rl_games/play.py) script for running a trained agent. I ran into a lot of version issues with different python packages due to Jetbot’s age so I didn’t get this to work. Instead I ended up exporting the trained model to the [onnx](https://onnx.ai/) format by adapting the [rl_games onnx export example](https://colab.research.google.com/github/Denys88/rl_games/blob/master/notebooks/train_and_export_onnx_example_continuous.ipynb) into this [onnx export script](https://github.com/ih/JetbotSim2RealBasic/blob/main/export_onnx.py). You can run it from the command line in a similar way to the training script.

`.\IsaacLab\isaaclab.bat -p .\JetbotSim2RealBasic\export_onnx.py --task Isaac-Jetbot-Direct-v0 --num_envs 1`

 I then copied the onnx file to Jetbot e.g.

`scp .\logs\rl_games\jetbot_direct\drl_forward_opset16.onnx jetbot@xxx.xxx.xx.xx:/home/jetbot`

On the Jetbot I adapted parts of the Jetbot [collision avoidance tutorial](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/notebooks/collision_avoidance/live_demo_resnet18_trt.ipynb) as well as the rl_games onnx export example to load the onnx model 

```
import onnxruntime as ort
import numpy as np
import torch

ort_model = ort.InferenceSession("drl_forward_opset16.onnx", providers=['CUDAExecutionProvider', 'TensorrtExecutionProvider'])
```


then feed it images from the camera and use the model output to drive the motors.

```
def update(change):
    image = change['new']
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0) 
    outputs = ort_model.run(None, {"obs": image})
    mu = torch.tensor(outputs[0][0])
    post_processed_mu = rescale_actions(-.1, .1, torch.clamp(mu, -1.0, 1.0))
    robot.set_motors(post_processed_mu[0].item(), post_processed_mu[1].item())
```

You can see the whole Jupyter notebook used on the Jetbot [here](https://github.com/ih/JetbotSim2RealBasic/blob/main/jetbot_onboard/drl_forward_onnx.ipynb).

There are a few things worth mentioning about the transfer of the model to the robot. The opset_version for the [onnx export](https://github.com/ih/JetbotSim2RealBasic/blob/main/export_onnx.py#L147) was set to 16 due to the onnxruntime that was on Jetbot

```torch.onnx.export(traced, *adapter.flattened_inputs, os.path.join(log_root_path, "drl_forward_opset16.onnx"), opset_version=16, input_names=['obs'], output_names=['mu','log_std', 'value'])```

I also ended up clipping the action values 1 to .1 since the first time I ran it the Jetbot moved too fast. I tried using robot.stop() to stop the robot, but it didn’t seem to work so I used `sudo shutdown now` from a terminal ssh’ed into the Jetbot. Here’s a video of the Jetbot using the model trained in simulation.

<iframe width="560" height="315" src="https://www.youtube.com/embed/1YCB_5nSyKY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


Even though the same output is being applied to left and right motors the Jetbot is initially turning because one of the wheels is loose and tilted. Once I manually adjust the wheel the Jetbot proceeds to move forward in a straight(-ish) line like it was trained to do, but it’s an example of the details in the real world that can be missed when modeling in simulation.

## Conclusion and Next Steps

So there you have it. A model trained in an Isaac Lab simulation running on a physical Jetbot in the real world. Now that I have a familiarity with the tools and process of going from simulation to reality, I’d like to try learning more difficult tasks. Especially ones where the vision input is critical and also seeing what it takes to get robustness in real world performance. I’d also like to learn more about scaling up to complex tasks by first learning simpler tasks like in the DeepMind humanoid soccer work. I hope this post makes the sim2real process easier for the next person who tries it and I look forward to solving more challenging problems and sharing what I learn.




