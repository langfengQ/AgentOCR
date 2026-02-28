<p align="center" style="margin-bottom: -10px;">
    <img src="./docs/agentocr/agentocr_logo.png" alt="logo" width="10%">
</p>
<h1 align="center">
AgentOCR: Reimagining Agent History via Optical Self-Compression
</h1>



<p align="center">
  <a href="https://arxiv.org/abs/2601.04786">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://huggingface.co/papers/2601.04786">
    <img src="https://img.shields.io/badge/HF%20Daily%20Paper-yellow?logo=huggingface" alt="HuggingFace Daily Paper"></a>
  &nbsp;
  <a href="https://github.com/langfengQ/AgentOCR">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?logo=github" alt="GitHub Project"></a>
  &nbsp;
  <a href="https://x.com/langfengq/status/2010609879665492420">
    <img src="https://img.shields.io/badge/Twitter-Channel-000000?logo=x" alt="X Channel"></a>
</p>


AgentOCR addresses the critical bottleneck of rapidly growing textual histories in multi-turn LLM agent training by representing observation-action history as **compact rendered images**. This approach exploits the superior information density of visual tokens, substantially reducing token consumption while preserving agent performance.

<p align="center">
    <img src="./docs/agentocr/agentocr_overview.png" alt="AgentOCR Overview" width="90%">
</p>

**Key Features:**
- **Visual Token Representation**: Renders history as compact images, achieving >50% token reduction with >95% performance preservation
- **Segment Optical Caching**: Hashable segment decomposition with visual cache enables 20× rendering speedup
- **Agentic Self-Compression**: Agent learns to adaptively emit compression rates via compression-aware reward training


# Installation
## Install veRL
```bash
conda create -n AgentOCR python==3.12 -y
conda activate AgentOCR

pip3 install vllm==0.11.0

pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip3 install -e .
```

## Install Supported Environments

### 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

### 2. Search
```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset (data will be saved at `~/data/searchR1_processed_direct`):
```bash
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```


Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration. Build Retriever environments:
```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install numpy==1.26.4 
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

Download the index:
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server: 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```

# Run Examples

We provide training scripts for ALFWorld and Search-based QA tasks:

```bash
# ALFWorld
bash train_alfworld.sh

# Search
bash train_search.sh
```

# Acknowledgement

AgentOCR is built upon [verl-agent](https://github.com/langfengQ/verl-agent) and [veRL](https://github.com/volcengine/verl), which provide the foundational infrastructure for multi-turn agent training and efficient RL training for LLMs. 

The supported environments are adapted from [ALFWorld](https://github.com/alfworld/alfworld) for embodied AI tasks and [Search-R1](https://github.com/PeterGriffinJin/Search-R1) for search-based question answering. We extend our gratitude to the authors and contributors of these projects for their valuable work.
