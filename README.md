# VisMind: Stateful Synthetic Dialogue for Visual Mid-training

VisMind is a multi-agent data synthesis framework designed for the mid-training phase of Vision-Language Models (VLMs). 

To address the limitations of conventional stateless generation, VisMind introduces a stateful, interactive paradigm. It transforms static visual instruction data into multi-turn reasoning trajectories that encapsulate step-by-step logical derivations, bidirectional peer critique, and supervised self-correction.

## ⚙️ System Architecture

Built on [LangGraph](https://github.com/langchain-ai/langgraph), the framework orchestrates a controlled Markovian interaction among three specialized agents:
* **Student Alpha & Beta (Peer Agents):** Engage in multi-turn collaborative problem-solving. They propose hypotheses, challenge each other's logic, and actively explore alternative reasoning paths.
* **Teacher Agent:** Possesses privileged access to the Ground-Truth solution. When invoked, it evaluates the students' reasoning and provides constructive hints and corrective feedback without directly revealing the final answer.

## 📂 Repository Structure

```text
VisMind/
├── agents/                  # Core multi-agent logic
│   ├── cores/               # Agent nodes and state definitions
│   ├── utils/               # Prompt templates and routing helpers
│   ├── students_teacher.py  # The primary 3-agent (S-S-T) graph builder
│   └── generator_supervisor.py # Alternative 2-agent graph builder
├── tools/                   # Dataset processing and utility scripts
│   ├── make_pt_dataset.py   # Formats data for Continued Pretraining (CPT)
│   ├── make_sft_dataset.py  # Formats data for Supervised Fine-Tuning (SFT)
│   ├── token_counter.py     # Counts total and average tokens
│   ├── token_dedupor.py     # Deduplicates samples based on image paths
│   └── token_truncator.py   # Truncates dataset based on a maximum token budget
├── app.py                   # Main entry point for parallel data synthesis
├── config.py                # Global configurations (model paths, concurrency)
├── processor.py             # Single-file processing and LangGraph execution
└── server.py                # Automated lifecycle management for vLLM servers
```

## 🛠️ Setup & Configuration

### 1. Environment Requirements
Ensure you have a dedicated Conda environment installed with the necessary dependencies:
```bash
pip install langchain-core langchain-openai langgraph vllm tiktoken psutil python-dotenv requests
```

### 2. Environment Variables
Create a `.env` file in the root directory to configure your OpenAI-compatible API endpoints. 
```env
STUDENT_MODEL_BASE="http://127.0.0.1:8000/v1"
STUDENT_MODEL_KEYS="EMPTY"
TEACHER_MODEL_BASE="http://127.0.0.1:8000/v1"
TEACHER_MODEL_KEYS="EMPTY"
```

### 3. Source Code Configuration (Action Required)
Before executing the pipeline, **you must update the hardcoded paths** in `config.py` to match your local file system:
* `CONDA_ENV_PATH`: Path to your Conda environment (used by `server.py` to auto-launch vLLM).
* `STUDENT_MODEL_PATH` & `TEACHER_MODEL_PATH`: Absolute paths to the local model weights.
* `CONCURRENCY`: Adjust the `ThreadPoolExecutor` max workers (default: 80) based on your CPU/RAM capacity.

## 🚀 Usage Guide

### Step 1: Prepare Raw Data
Place your raw instruction data inside the `input/` directory. The framework expects individual JSON files in a designated sub-folder. Each JSON file must contain:
* `question`: The textual query.
* `answer`: The ground-truth answer.
* `image_paths`: A string or a list of strings representing the image path(s).

### Step 2: Run Data Synthesis
Execute `app.py` to start the multi-agent synthesis pipeline. 

> ⚠️ **Hardware & Server Assumption:** > `app.py` automatically spins up a local vLLM server for the **Teacher** agent (defaulting to `devices=[0, 1]`, `tensor_parallel_size=2`). 
> **You must independently start and manage the vLLM server for the Student model** ensuring it is accessible via the URL specified in your `.env` file.

```bash
python app.py \
    --input_dir ./input/VisualWebInstruct118K \
    --output_dir ./output/VisualWebInstruct118K/event-0608-01 \
    --json_folder json
```

### Step 3: Build Training Datasets
Once synthesis is complete, utilize the scripts in `tools/` to filter the raw event logs into training-ready corpora.

> ⚠️ **Path Modification:** All scripts in the `tools/` directory contain hardcoded input/output paths within their `if __name__ == "__main__":` blocks. **Manually edit these paths before execution.**

* **Build CPT (Mid-training) Dataset:**
  Executes rule-based filtering (language verification, stripping routing markers like `#TO_TEACHER#`, removing triple-word repetitions) and concatenates valid dialogues:
  ```bash
  python tools/make_pt_dataset.py
  ```

* **Build SFT Dataset:**
  Extracts original QA pairs mapped from the validated IDs, adhering to a strict token length limit (default: 4096):
  ```bash
  python tools/make_sft_dataset.py
  ```

### Step 4: Auxiliary Tools
* **Token Counting:** `python tools/token_counter.py` (Calculates individual and combined token statistics using `tiktoken`).
* **Deduplication:** `python tools/token_dedupor.py` (Removes duplicate visual queries sharing the exact same first image).
* **Truncation/Sampling:** `python tools/token_truncator.py` (Randomly samples the dataset to fit within a specified maximum token budget).
