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

SEARCH_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct reasoning process about the question and the information in the image.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you lack some knowledge (including any uncertainty, missing detail, or need to verify facts), you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough information to answer the question confidently, provide your final answer within <answer> </answer> tags. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct reasoning process about the question and the information in the image.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you lack some knowledge (including any uncertainty, missing detail, or need to verify facts), you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough information to answer the question confidently, provide your final answer within <answer> </answer> tags. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE_NO_HIS_OCR = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

The image below will be used to record all subsequent search queries and results as you progress through the task. Initially it is blank, but it will be updated to visualize your interaction history.
<image>

Now it's your turn to respond for the current step.
You should first conduct reasoning process about the question and the information in the image.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you lack some knowledge (including any uncertainty, missing detail, or need to verify facts), you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough information to answer the question confidently, provide your final answer within <answer> </answer> tags. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE_OCR = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). The image below demonstrates the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History: <image>

Now it's your turn to respond for the current step.
You should first conduct reasoning process about the question and the information in the image.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you lack some knowledge (including any uncertainty, missing detail, or need to verify facts), you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough information to answer the question confidently, provide your final answer within <answer> </answer> tags. For example, <answer>Beijing</answer>.
"""

SEARCH_COMPRESSION_TEMPLATE_NO_HIS = """
Additionally, you need to select an image compression factor (> 1.0) for the next history image. Higher compression reduces cost, but over-compression degrades image quality and can lower task success rates. Therefore, you should select the highest compression level that preserves essential information for reliable task completion. You must present your next compression factor within <compression></compression> tags (e.g., <compression>1.1</compression>).
"""

SEARCH_COMPRESSION_TEMPLATE = """
Additionally, you need to select an image compression factor (> 1.0) for the next history image (note: the above provideded image uses a compression factor of {compression_factor}). Higher compression reduces cost, but over-compression degrades image quality and can lower task success rates. Therefore, you should select the highest compression level that preserves essential information for reliable task completion. You must present your next compression factor within <compression></compression> tags (e.g., <compression>1.1</compression>).
"""