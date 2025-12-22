# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_NO_HIS_OCR = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current textual observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

The image below will be used to record all subsequent observations and actions as you progress through the task. Initially it is blank, but it will be updated to visualize your interaction history.
<image>

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_OCR = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s). The image below demonstrates the most recent {history_length} observations and the corresponding actions you took:
<image>

You are now at step {current_step} and your current textual observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_COMPRESSION_TEMPLATE_NO_HIS = """
Additionally, you need to select an image compression factor (> 1.0) for the next history image. Higher compression reduces cost, but over-compression degrades image quality and can lower task success rates. Therefore, you should select the highest compression level that preserves essential information for reliable task completion. You must present your next compression factor within <compression></compression> tags (e.g., <compression>1.1</compression>).
"""

ALFWORLD_COMPRESSION_TEMPLATE = """
Additionally, you need to select an image compression factor (> 1.0) for the next history image (note: the above provideded image uses a compression factor of {compression_factor}). Higher compression reduces cost, but over-compression degrades image quality and can lower task success rates. Therefore, you should select the highest compression level that preserves essential information for reliable task completion. You must present your next compression factor within <compression></compression> tags (e.g., <compression>1.1</compression>).
"""