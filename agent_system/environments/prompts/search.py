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
You should first conduct a reasoning process about the question and what is still missing, uncertain, ambiguous, or unverified. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history, where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process about the question, what is already known from previous <information>, and what is still missing, uncertain, ambiguous, or unverified. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE_NO_HIS_OCR = """<image>
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct a reasoning process about the question and what is still missing, uncertain, ambiguous, or unverified. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE_OCR = """<image>
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). The provided image shows the full interaction history so far, where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine.

Now it's your turn to respond for the current step.
You should first conduct a reasoning process about the question, what is already known from previous <information>, and what is still missing, uncertain, ambiguous, or unverified. After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
(2) Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags. For example, <answer>Beijing</answer>.
"""

SEARCH_COMPRESSION_TEMPLATE_NO_HIS = """
Additionally, you need to select an image compression factor (>1.0) for the next image. Higher compression reduces cost, but over-compression degrades image quality and obscures the interaction history. Thus, you should select the highest compression level that preserves essential information for reliable task completion. You must present your next compression factor within <compression></compression> tags (e.g., <compression>1.2</compression>).
"""

SEARCH_COMPRESSION_TEMPLATE = """
Additionally, you need to select an image compression factor (>1.0) for the next image (note: the provided image uses a compression factor of {compression_factor}). Higher compression reduces cost, but over-compression degrades image quality and obscures the interaction history. Thus, you should select the highest compression level that preserves essential information for reliable task completion. You must present your next compression factor within <compression></compression> tags (e.g., <compression>1.2</compression>).
"""