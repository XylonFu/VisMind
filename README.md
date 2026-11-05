# VisMind: Stateful Synthetic Dialogue for Visual Mid-Training

**Xinlong Fu · Zihao Zhou · Qiufeng Wang**  
*Findings of the Association for Computational Linguistics: EMNLP 2026*

[📄 Paper](https://aclanthology.org/2026.findings-emnlp.1158/) | [📥 PDF](https://aclanthology.org/2026.findings-emnlp.1158.pdf) | [💻 Code](https://github.com/XylonFu/VisMind)

VisMind is a multi-agent data synthesis framework designed for the mid-training phase of Vision-Language Models (VLMs).

Addressing the limitations of conventional stateless synthesis, VisMind introduces a stateful, interactive paradigm. It transforms static visual instruction seeds into complex, multi-turn reasoning trajectories characterized by iterative problem-solving, bidirectional peer critique, and supervised intervention.

## ⚙️ Core Mechanism

Built on [LangGraph](https://github.com/langchain-ai/langgraph), the framework orchestrates a multi-turn Markovian interaction among three specialized agents:

- **Peer-reasoning Students ($\alpha$ and $\beta$):** Engage in bidirectional critique. They iteratively generate reasoning steps, explore alternative paths, and validate or challenge each other's logic to encourage autonomous revisions.
- **Ground-truth-aware Teacher ($\tau$):** Possesses privileged access to the ground-truth solution. When invoked, it evaluates the students' accumulated dialogue history and provides constructive hints and corrective feedback without directly revealing the final answer.

## 📂 Repository Structure

```text
VisMind/
├── agents/                     # Core multi-agent logic
│   ├── cores/                  # Agent nodes and state definitions
│   ├── utils/                  # Prompt templates and routing helpers
│   ├── students_teacher.py     # Primary 3-agent (S-S-T) graph builder
│   └── generator_supervisor.py # Alternative 2-agent graph builder
├── tools/                      # Dataset processing and utility scripts
│   ├── make_pt_dataset.py      # Rule-based filtering for Continued Pretraining (CPT)
│   ├── make_sft_dataset.py     # Formats original seeds for Supervised Fine-Tuning (SFT)
│   ├── token_counter.py        # Counts total and average tokens
│   ├── token_dedupor.py        # Deduplicates samples based on image paths
│   └── token_truncator.py      # Truncates data to a maximum token budget
├── app.py                      # Main entry point for parallel data synthesis
├── config.py                   # Global configuration (model paths, concurrency)
├── processor.py                # Single-file processing and LangGraph execution
└── server.py                   # Automated lifecycle management for vLLM servers
```

## 🛠️ Setup & Configuration

### 1. Environment Requirements

Clone the repository and install the required dependencies in a dedicated Conda environment:

```bash
git clone https://github.com/XylonFu/VisMind.git
cd VisMind

pip install langchain-core langchain-openai langgraph vllm tiktoken psutil python-dotenv requests
```

### 2. Environment Variables

Create a `.env` file in the repository root to configure the OpenAI-compatible API endpoints:

```env
STUDENT_MODEL_BASE="http://127.0.0.1:8001/v1"
STUDENT_MODEL_KEYS="EMPTY"
TEACHER_MODEL_BASE="http://127.0.0.1:8000/v1"
TEACHER_MODEL_KEYS="EMPTY"
```

> **Note:** The example assigns separate ports to the student (`8001`) and teacher (`8000`) services to avoid port conflicts. Update these values to match your actual deployments. The student vLLM server must be started independently; the teacher server is managed by `app.py`.

### 3. Source Code Configuration

Before running the pipeline, update the hardcoded paths in `config.py` to match your local environment:

- **`CONDA_ENV_PATH`:** Path to your Conda environment (used by `server.py` to launch vLLM).
- **`STUDENT_MODEL_PATH` and `TEACHER_MODEL_PATH`:** Absolute paths to the corresponding local model weights.
- **`CONCURRENCY`:** Number of `ThreadPoolExecutor` workers (default: `80`); adjust based on your hardware capacity.

Ensure the student and teacher model services have sufficient GPU resources. By default, `app.py` allocates GPUs `0` and `1` to the teacher server.

## 🚀 Usage Guide

### Step 1: Seed Data Preparation

Place your raw visual instruction data inside the `input/` directory. The framework expects individual JSON files in a designated subfolder. Each JSON instance must contain:

- **`question`:** The multimodal query.
- **`answer`:** The static ground-truth answer.
- **`image_paths`:** A string or a list of strings representing the image path(s).

### Step 2: SSD Generation

Run `app.py` to start the multi-agent synthesis pipeline:

> ⚠️ **Hardware and server dependency:** `app.py` automatically launches a local vLLM server for the teacher agent (default: `devices=[0, 1]`, `tensor_parallel_size=2`, port `8000`). You must start and manage the student vLLM server separately, using the address configured in `.env` (port `8001` in the example above).

```bash
python app.py \
    --input_dir ./input/VisualWebInstruct118K \
    --output_dir ./output/VisualWebInstruct118K/event-0608-01 \
    --json_folder json
```

### Step 3: Data Filtering & Pipeline Preparation

Use the scripts in `tools/` to transform the raw interaction traces into training-ready datasets.

> ⚠️ **Path modification required:** The scripts in `tools/` contain hardcoded input/output paths in their `if __name__ == "__main__":` blocks. Update these paths before execution.

**Build the CPT (mid-training) dataset**

Validates the terminal marker (`#END_CONVERSATION#`) and applies three stages of rule-based filtering: language verification, repetition detection, and format cleaning. The result is the filtered Stateful Synthetic Dialogue (SSD) corpus.

```bash
python tools/make_pt_dataset.py
```

**Build the SFT dataset**

Extracts the original visual instruction seeds and formats them into standard question-answer pairs under a token length limit (default: `4096`).

```bash
python tools/make_sft_dataset.py
```

### Step 4: Auxiliary Tools

- **Token counting:** `python tools/token_counter.py` — Calculates individual and aggregate token statistics using `tiktoken`.
- **Deduplication:** `python tools/token_dedupor.py` — Removes duplicate visual queries sharing the same first image path.
- **Truncation/sampling:** `python tools/token_truncator.py` — Randomly samples the dataset to fit within a maximum token budget.

## 📖 Citation

If you find VisMind useful in your research, please consider citing our paper:

```bibtex
@inproceedings{fu2026vismind,
  title={VisMind: Stateful Synthetic Dialogue for Visual Mid-Training},
  author={Fu, Xinlong and Zhou, Zihao and Wang, Qiufeng},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages={24493--24510},
  year={2026},
  url={https://aclanthology.org/2026.findings-emnlp.1158/},
  doi={10.18653/2026.findings-emnlp.1158}
}
```
