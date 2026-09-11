# 第二步：Qwen 0.5B LoRA 微调与评估

[第一步：环境准备与从零预训练](./01-环境与预训练.md) · [第三步：PC 部署与手机扩展](./03-端侧部署.md)

目标是在本机 RTX 3050 Laptop 4GB 显存上完成一轮有监督微调，而不是全参数训练 Qwen。

下面给出可保存成文件的完整教学示例，API 按指定版本编写；已做静态检查，不代表已在这张 GPU 上执行。普通 FP16 LoRA 仍是第一选择，因为依赖更简单；先以最大长度 128 做 10 步冒烟测试，通过后完成第一轮正式实验。只有实测显存不足，才引入 bitsandbytes QLoRA。

开始前重新执行第一步的资源预检：系统可用内存建议至少约 6GiB，`nvidia-smi` 中不要留有 Ollama 或其他计算进程，笔记本接通电源并切到性能模式。本机虽然装有 Docker、存在 WSL 程序，但第一轮不使用它们，以免同时引入操作系统和容器变量。

## 1. 创建微调环境并下载模型

新开 PowerShell。沿用上一篇的驱动与 GPU 验收条件，创建独立环境：

```powershell
Set-Location D:\workspace\pretrain-model
$PROJECT_ROOT = (Get-Location).Path
py -3.12 -m venv .\.venv-finetune
& .\.venv-finetune\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install transformers==4.56.2 peft==0.17.1 accelerate==1.10.1 datasets==4.0.0
python -m pip check
$env:HF_HOME = Join-Path $PROJECT_ROOT "cache\huggingface"
python .\check_gpu.py
New-Item -ItemType Directory -Force .\finetune\data
Set-Location .\finetune
```

采用 [Transformers Trainer 4.56.2 文档](https://huggingface.co/docs/transformers/v4.56.2/en/trainer)对应的 API 和 [PEFT 0.17 系列的 LoRA 工作流](https://huggingface.co/docs/peft/v0.17.0/en/quicktour)。这些是教学版本基线，安装成功后再保存 `pip freeze`；不要把本章依赖装入 nanoGPT 环境。

将下面代码保存为 `download_model.py` 并运行 `python download_model.py`：

```python
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
repo = "Qwen/Qwen2.5-0.5B-Instruct"
revision = HfApi().model_info(repo).sha
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=ROOT / "models" / "qwen-base",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
Path("base-revision.txt").write_text(f"{repo}\n{revision}\n", encoding="utf-8")
print("模型已下载；请保留 base-revision.txt 和整个模型目录")
```

下载的 `qwen-base` 名称仅表示“本实验的基座目录”，内容实际是 **Instruct** 模型。后续训练、评估和合并全部使用这个目录，避免加载到不同权重版本。若模型站点网络不可达，先解决下载问题；不要把失败误判为显卡或训练故障。

## 2. 选一个可以量化评估的小任务

推荐“客户留言 → 固定 JSON 意图”，四类：`退款`、`物流`、`商品咨询`、`其他`。它输出短、标注简单，能观察格式遵循和分类准确率。复杂聊天风格很难用第一次实验判断是否变好。

先定义标注规则：询问或要求退钱为退款；快递位置、送达时间为物流；商品规格、用途为商品咨询；其余为其他。多意图样本第一轮可以排除，后续再写优先级规则。

在 `data/train.jsonl` 中按以下格式保存，每行一个 JSON 对象，UTF-8 编码：

```jsonl
{"instruction":"将客户留言分类为退款、物流、商品咨询、其他之一，只输出包含 intent 字段的 JSON。留言：取消订单以后钱什么时候退回来？","output":"{\"intent\":\"退款\"}"}
{"instruction":"将客户留言分类为退款、物流、商品咨询、其他之一，只输出包含 intent 字段的 JSON。留言：包裹已经到哪个城市了？","output":"{\"intent\":\"物流\"}"}
{"instruction":"将客户留言分类为退款、物流、商品咨询、其他之一，只输出包含 intent 字段的 JSON。留言：这件衣服是什么面料？","output":"{\"intent\":\"商品咨询\"}"}
{"instruction":"将客户留言分类为退款、物流、商品咨询、其他之一，只输出包含 intent 字段的 JSON。留言：谢谢你的帮助。","output":"{\"intent\":\"其他\"}"}
```

这四条只是格式示例，不足以证明学到了任务。你可以先准备训练 16 条、验证 8 条、测试 8 条做流程检查；正式实验建议至少 **训练 200 条、验证 40 条、测试 40 条**，各类尽量均衡，逐步增加到 500 条。

分别保存为 `data/train.jsonl`、`data/val.jsonl`、`data/test.jsonl`，三者格式相同。不要把上述四句简单改几个字就分别放进三个集合：同一模板、同一原始问题的改写要放在同一集合，避免数据泄漏。训练集用于学习；验证集用于选择学习率/epoch/checkpoint；测试集在确定方案之后用于最终评估。

标注质量优先于条数。不要用不一致的标签硬凑数据；数据来源和使用许可也记入实验记录。练习数据可自行编写，但需要明确它不能代表真实业务分布。

## 3. 先建立基座评估，训练后复用

保存为 `evaluate.py`：

```python
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "qwen-base"))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", default="data/val.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True,
    ).to("cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    assert rows, "评估集为空"
    results = []
    for row in rows:
        messages = [{"role": "user", "content": row["instruction"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        assert inputs.input_ids.shape[1] <= 128, "第一轮请使用短输入，保持评估条件一致"
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=48,
                                    pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = json.loads(row["output"])
        try:
            parsed = json.loads(answer)
            valid = isinstance(parsed, dict) and set(parsed) == {"intent"} and parsed["intent"] in {"退款", "物流", "商品咨询", "其他"}
            correct = valid and parsed == expected
        except (json.JSONDecodeError, TypeError):
            valid, correct = False, False
        results.append({"instruction": row["instruction"], "expected": expected,
                        "answer": answer, "valid_schema": bool(valid), "correct": bool(correct)})
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")
    print("samples:", len(results))
    print("schema_valid_rate:", sum(r["valid_schema"] for r in results) / len(results))
    print("task_accuracy:", sum(r["correct"] for r in results) / len(results))

if __name__ == "__main__":
    main()
```

先在验证集上跑基座，保留每条输出：

```powershell
python evaluate.py --data data/val.jsonl --out baseline-val.jsonl
```

不要为了把结果做漂亮不断修改测试集。本任务对原模型可能已经很简单，若基座接近满分，先明确实验已缺乏提升空间，增加真实且规则清晰的边界样本，不要声称训练后必然更好。

## 4. LoRA 训练教学代码

保存为 `train_lora.py`。该示例仅处理“一条用户输入、一条助手答案”的样本；复杂多轮数据需要另外设计标签遮罩。

```python
import argparse
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments, set_seed,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
BASE = str(ROOT / "models" / "qwen-base")

def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    assert rows, f"{path} 为空"
    for row in rows:
        assert isinstance(row["instruction"], str) and row["instruction"].strip()
        assert isinstance(row["output"], str) and row["output"].strip()
        obj = json.loads(row["output"])
        assert isinstance(obj, dict) and set(obj) == {"intent"}
        assert obj["intent"] in {"退款", "物流", "商品咨询", "其他"}
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/lora-v1")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    set_seed(42)
    train_rows, val_rows = read_rows("data/train.jsonl"), read_rows("data/val.jsonl")
    # 只能抓住完全重复，模板改写泄漏仍需人工检查。
    assert not ({r["instruction"] for r in train_rows} & {r["instruction"] for r in val_rows}), "训练/验证输入重复"
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def encode(rows):
        encoded = []
        for row in rows:
            user = [{"role": "user", "content": row["instruction"]}]
            prompt = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                user + [{"role": "assistant", "content": row["output"]}],
                tokenize=False, add_generation_prompt=False,
            )
            prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            assert ids[:len(prefix)] == prefix, "模板/tokenizer 前缀不一致，不能直接按长度 mask"
            assert len(ids) <= args.max_length, "样本超长：请缩短或单独处理，不要截掉整个答案"
            assert len(ids) > len(prefix), "没有可训练的答案 token"
            encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                            "labels": [-100] * len(prefix) + ids[len(prefix):]})
        return Dataset.from_list(encoded)

    train_data, val_data = encode(train_rows), encode(val_rows)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none",
    ))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    smoke = args.max_steps > 0
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4, num_train_epochs=args.epochs, max_steps=args.max_steps,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", max_grad_norm=1.0, warmup_ratio=0.1,
        logging_steps=1 if smoke else 5,
        eval_strategy="steps" if smoke else "epoch",
        save_strategy="steps" if smoke else "epoch",
        eval_steps=5, save_steps=5,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, dataloader_num_workers=0,
        report_to="none", seed=42, label_names=["labels"],
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_data, eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8,
        ),
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(Path(args.out) / "adapter"))
    tokenizer.save_pretrained(str(Path(args.out) / "adapter"))
    print("best checkpoint:", trainer.state.best_model_checkpoint)
    print("peak allocated GiB:", torch.cuda.max_memory_allocated() / 1024**3)
    print("peak reserved GiB:", torch.cuda.max_memory_reserved() / 1024**3)

if __name__ == "__main__":
    main()
```

这里把用户提示的标签设为 `-100`，只对助手答案与结束标记等答案后缀计算损失；padding 也设为 `-100`。CausalLM 会在内部完成预测位置与标签的错位处理，不要再手动 shift 一遍。动态 padding 避免每条短数据都无意义地补到 256。

LoRA 只加到 `q_proj`、`v_proj`，是为第一轮控制开销的选择，不保证最优。可训练参数占比应很小且不为零。不要用 `device_map="auto"` 把训练悄悄拆到 CPU；先明确用单张 CUDA 卡。

运行前用 `ollama ps` / `ollama stop 模型名` 卸载其他模型，结束上一次评估进程，然后执行：

```powershell
python train_lora.py --max-steps 10 --max-length 128 --out runs/smoke-128
python train_lora.py --epochs 3 --max-length 128 --out runs/lora-v1
python -m pip freeze > requirements-finetune-resolved.txt
```

第一条成功的条件：loss 为有限数值、至少一次验证和保存、适配器文件存在、无显存错误。第二条为正式训练，使用全新输出目录，从基座重新开始，不会自动沿用 smoke 的权重。第一轮任务的答案很短，128 已足够；不要仅为追求更大的数字先升到 256。

需要较长输入时，先单独运行 `python train_lora.py --max-steps 10 --max-length 256 --out runs/smoke-256`。只有这个冒烟测试通过，才用新的输出目录正式训练 256 长度，并分别记录峰值显存和步耗时。

例如训练集 200 条，batch 1、梯度累积 8，每个 epoch 约 25 次优化器更新，3 个 epoch 约 75 步。数据迭代次数、micro-batch 数和 optimizer step 不是同一个概念。

## 5. 重新加载、对比与恢复

```powershell
python evaluate.py --adapter runs/lora-v1/adapter --data data/val.jsonl --out lora-val.jsonl
```

确定配置后，分别对**同一个未参与调参的测试集**运行基座和适配器：

```powershell
python evaluate.py --data data/test.jsonl --out baseline-test.jsonl
python evaluate.py --adapter runs/lora-v1/adapter --data data/test.jsonl --out lora-test.jsonl
```

记录 schema 合格率、任务准确率以及失败样本。无效 JSON 按任务失败计入分母；不要只在“能解析的回答”里计算准确率。验证 loss 下降不等于任务准确率一定提升，也不等于获得了通用知识能力。

最终 `adapter/` 中通常有 `adapter_config.json`、`adapter_model.safetensors` 和 tokenizer 文件。它没有完整基座权重，不能单独当完整模型运行。[PEFT 官方指南](https://huggingface.co/docs/peft/v0.17.0/en/quicktour)说明了这种保存和加载方式。

需要恢复中断的训练时，先找到 `runs/lora-v1/checkpoint-*` 实际存在的目录，使用原数据、原配置和原输出目录：

```powershell
python train_lora.py --epochs 3 --out runs/lora-v1 --resume runs/lora-v1/checkpoint-25
```

`checkpoint-25` 是示例，必须替换为实际目录。完整训练 checkpoint 和最终 adapter 用途不同：前者用于恢复优化器等训练状态，后者用于加载微调结果。完成一轮训练后想继续更多 epoch，需要增加总 epoch 上限并记录此变化。

## 6. 显存不足时的调整顺序

先确认 OOM 是加载时、前向/反向时还是评估时发生。`nvidia-smi` 可观察进程；PyTorch 打印的是自身 allocated/reserved 峰值，不包含所有驱动和其他应用显存。

1. 停止 Ollama 已加载模型，关闭其他 GPU 程序，结束失败的训练进程再重跑。
2. 本示例 batch 已为 1、第一轮长度已为 128；继续缩短输入和答案，但必须保留完整监督答案。仍不足就改用更小模型，不要把答案静默截断。
3. 确认梯度检查点已开启、训练时 `use_cache=False`、评估只保留 loss。
4. 仍不够，再尝试同一模型的 QLoRA，或更小模型；第一轮不要扩大到 1.5B/7B。

增加梯度累积不能减少 batch=1 时单次前向的显存。降低 LoRA rank 可能有帮助，但未必解决主要由激活造成的 OOM。

### 可选 QLoRA 改法

QLoRA 是“低位量化基座 + LoRA”。可在微调环境安装 `bitsandbytes`，用 NF4 4bit 加载基座，先调用 `prepare_model_for_kbit_training` 再调用 `get_peft_model`；计算精度仍然使用 FP16。量化不一定覆盖所有参数，也不会消除激活开销。

下面片段替换训练脚本中加载基座的部分，后续保留 LoRA 创建与 Trainer 设置；它是备选改法，不要和原 FP16 加载同时执行：

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map={"": 0}, torch_dtype=torch.float16,
    attn_implementation="sdpa", local_files_only=True,
)
model = prepare_model_for_kbit_training(model)
```

运行 `python -m pip install bitsandbytes` 和 `python -m bitsandbytes` 检查环境，并保存实际安装版本。官方文档目前为 Windows x86-64 提供包含 `sm86` 的 CUDA wheel，NF4/FP4 要求的计算能力也低于本机 8.6；因此硬件条件匹配，但具体 Python、PyTorch、CUDA runtime 与 wheel 组合仍必须实测，不能把“理论支持”写成“已经跑通”。参考 [bitsandbytes 安装说明](https://huggingface.co/docs/bitsandbytes/en/installation)和 [PEFT 量化指南](https://huggingface.co/docs/peft/en/developer_guides/quantization)。

如果 Windows 上出现量化库兼容问题，先保留已经完成的普通 LoRA 小实验；需要时再单独迁移 WSL2。不要同时切换操作系统、模型和数据，否则难以定位问题。

QLoRA 的合并同样从原始未量化基座加载 adapter，再进行合并、重新量化；量化训练和部署量化并非完全相同的数值路径，需要重新评估。

保存 adapter、基座与微调后的同集评估结果后，进入[第三步：PC 部署与手机扩展](./03-端侧部署.md)。
