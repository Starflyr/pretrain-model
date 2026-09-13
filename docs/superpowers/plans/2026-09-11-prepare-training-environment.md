# Training Environment Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare this Windows laptop for the repository's nanoGPT pretraining exercise without starting training.

**Architecture:** Use native Windows PowerShell with Python 3.12. Keep pretraining and fine-tuning dependencies in separate virtual environments under the repository. Store all source code, caches, model weights, generated datasets, and revision records below `D:\workspace\pretrain-model`.

**Tech Stack:** Python 3.12, PyTorch 2.8.0 CUDA 12.6 wheel, NumPy, Requests, tiktoken, nanoGPT.

**Spec:** `learning-guide-4gb/01-环境与预训练.md` and `learning-guide-4gb/02-LoRA微调.md`

**Scope update (2026-09-11):** At the user's request, Tasks 4 and 5 are deferred. No fine-tuning environment, Transformers/PEFT packages, Qwen model, or LoRA JSONL data will be installed or downloaded in this execution.

## Global Constraints

- Run natively on Windows; do not introduce WSL, Docker, Conda, DeepSpeed, FlashAttention, or Ollama.
- Keep Codex, ChatGPT, Clash, NVIDIA display services, audio services, and Huorong running.
- Do not start nanoGPT training, LoRA training, baseline model evaluation, or local model serving.
- Use `D:\workspace\pretrain-model` as the project root for every created artifact and cache.
- Use the pinned package versions from the guides and verify the CUDA forward/backward path on the RTX 3050 Laptop GPU.

---

### Task 1: Install Python 3.12

**Files:**
- Create: `.downloads/python-3.12-amd64.exe`

**Interfaces:**
- Consumes: official Python Windows installer URL and checksum metadata.
- Produces: a callable 64-bit Python 3.12 interpreter and launcher.

- [x] **Step 1: Download the official current Python 3.12 x64 installer**

Download it into `.downloads`, then calculate its SHA-256 hash and compare it with Python.org metadata.

- [x] **Step 2: Install Python for the current user**

Run the installer silently with pip and the Python launcher enabled, without changing unrelated machine settings.

- [x] **Step 3: Verify the interpreter**

Run `py -3.12 --version`, `py -3.12 -c "import struct; print(struct.calcsize('P') * 8)"`, and `py -3.12 -m pip --version`; expect Python 3.12, 64, and a working pip.

### Task 2: Prepare and verify the pretraining environment

**Files:**
- Create: `.venv-pretrain/`
- Create: `cache/pip/`
- Create: `cache/huggingface/`
- Create: `check_gpu.py`
- Create: `requirements-pretrain-resolved.txt`

**Interfaces:**
- Consumes: Python 3.12 and NVIDIA driver 576.83.
- Produces: `.venv-pretrain\Scripts\python.exe` with CUDA-enabled PyTorch and a GPU verification record.

- [x] **Step 1: Create the isolated environment**

Run `py -3.12 -m venv .venv-pretrain`.

- [x] **Step 2: Install pinned pretraining dependencies**

Run `.venv-pretrain\Scripts\python.exe -m pip install --upgrade pip`, install `torch==2.8.0` from `https://download.pytorch.org/whl/cu126`, then install `numpy requests tiktoken`.

- [x] **Step 3: Verify dependency integrity and CUDA**

Run `pip check`, then execute `check_gpu.py`; require CUDA available, RTX 3050 detection, and a successful GPU forward/backward pass.

- [x] **Step 4: Freeze resolved versions**

Run `pip freeze` and save it to `requirements-pretrain-resolved.txt`.

### Task 3: Download nanoGPT and prepare Shakespeare data

**Files:**
- Create: `nanoGPT/`
- Create: `nanogpt-revision.txt`
- Create: `nanoGPT/data/shakespeare_char/input.txt`
- Create: `nanoGPT/data/shakespeare_char/train.bin`
- Create: `nanoGPT/data/shakespeare_char/val.bin`
- Create: `nanoGPT/data/shakespeare_char/meta.pkl`

**Interfaces:**
- Consumes: Git, `.venv-pretrain`, and the nanoGPT repository.
- Produces: pinned nanoGPT source and encoded train/validation data ready for the documented smoke test.

- [x] **Step 1: Clone nanoGPT**

Run `git clone https://github.com/karpathy/nanoGPT.git nanoGPT` and write `git -C nanoGPT rev-parse HEAD` to `nanogpt-revision.txt`.

- [x] **Step 2: Prepare the character dataset**

Run `.venv-pretrain\Scripts\python.exe nanoGPT\data\shakespeare_char\prepare.py` with `nanoGPT` as the working directory.

- [x] **Step 3: Verify outputs**

Require non-empty `train.bin`, `val.bin`, and `meta.pkl`; load `meta.pkl` using the isolated interpreter and verify `vocab_size`, `stoi`, and `itos` are present.

### Task 4: Prepare the LoRA environment and base model (deferred)

**Files:**
- Create: `.venv-finetune/`
- Create: `models/qwen-base/`
- Create: `finetune/download_model.py`
- Create: `finetune/base-revision.txt`
- Create: `requirements-finetune-resolved.txt`

**Interfaces:**
- Consumes: Python 3.12, Hugging Face, and `Qwen/Qwen2.5-0.5B-Instruct`.
- Produces: pinned local model files and an isolated LoRA environment.

- [ ] **Step 1: Create and populate the isolated environment**

Create `.venv-finetune`; install pip, PyTorch 2.8.0 CUDA 12.6, `transformers==4.56.2`, `peft==0.17.1`, `accelerate==1.10.1`, and `datasets==4.0.0`.

- [ ] **Step 2: Verify packages and CUDA**

Run `pip check` and `check_gpu.py` with `.venv-finetune\Scripts\python.exe`.

- [ ] **Step 3: Download a fixed model revision**

Resolve the current repository commit using `HfApi().model_info(repo).sha`, download only JSON, safetensors, TXT, and tokenizer model files into `models/qwen-base`, and write the exact repository and revision to `finetune/base-revision.txt`.

- [ ] **Step 4: Verify the offline model snapshot**

Load the tokenizer and model configuration with `local_files_only=True`, verify safetensors files are present and non-empty, then save `pip freeze` to `requirements-finetune-resolved.txt`.

### Task 5: Create and validate the teaching dataset (deferred)

**Files:**
- Create: `finetune/data/train.jsonl`
- Create: `finetune/data/val.jsonl`
- Create: `finetune/data/test.jsonl`
- Create: `finetune/validate_data.py`

**Interfaces:**
- Consumes: the four-label schema from `02-LoRA微调.md`.
- Produces: deterministic UTF-8 JSONL splits with 200 training, 40 validation, and 40 test examples.

- [ ] **Step 1: Generate disjoint examples**

Create balanced examples for `退款`, `物流`, `商品咨询`, and `其他`; keep every semantic template family in only one split so paraphrases do not leak across train, validation, and test.

- [ ] **Step 2: Validate schema and split integrity**

For every row, require exactly `instruction` and `output`; require `output` to parse as a one-key JSON object whose `intent` is one of the four labels; require no duplicate instruction across all splits and exactly 200/40/40 rows.

- [ ] **Step 3: Check token lengths**

Use the downloaded Qwen tokenizer and chat template to verify all first-round inputs and labels fit within the guide's 128-token maximum.

### Task 6: Final readiness report

**Files:**
- Modify: none.

**Interfaces:**
- Consumes: all prior task artifacts and command output.
- Produces: evidence that environments and data are ready while training remains stopped.

- [x] **Step 1: Run full readiness checks**

Verify the Python version, `pip check` in the pretraining environment, CUDA forward/backward, nanoGPT dataset artifacts, free disk space, GPU process state, and that no training Python process is running. Confirm the deferred fine-tuning environment and Qwen snapshot are absent.

- [x] **Step 2: Report exact revisions, sizes, and blockers**

Report Python/PyTorch/CUDA versions, nanoGPT commit, data sizes, remaining disk space, and any step that did not pass.
