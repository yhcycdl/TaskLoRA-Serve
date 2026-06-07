# TaskLoRA-Serve 实验执行总流程

这份文档告诉你整个项目从环境准备到最终报告填数，具体要做什么、跑什么命令、每一步产物在哪里。你可以把它当成项目执行 checklist。

## 0. 项目总目标

TaskLoRA-Serve V1 要完成一条完整的大模型工程链路：

```text
数据集下载与转换
-> QLoRA 训练 code/math 两个 adapter
-> base vs LoRA 质量评测
-> vLLM Multi-LoRA Serving
-> FastAPI Gateway task routing
-> mixed workload benchmark
-> Prometheus metrics
-> README/report 填真实实验结果
```

最终要能在简历里讲清楚：

- 我用了哪些公开数据集。
- 我怎么训练 task-specific LoRA adapter。
- LoRA 相比 base model 在目标任务上有没有提升。
- vLLM + Gateway 如何实现 task -> adapter 路由。
- 混合 workload 下延迟、吞吐、错误率、adapter 分布是多少。

## 1. 环境准备

训练和真实 vLLM serving 建议使用 Python 3.10 或 3.11。不要用 Python 3.13 跑训练。

```bash
cd /home/cyh/llm+infra

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The dependency versions are pinned for the CUDA 12.1 training stack. Do not run a loose `pip install -U transformers torch vllm`, because newer packages may require newer NVIDIA drivers.

如果你只想先跑本地 mock 演示，不需要装完整训练依赖：

```bash
pip install -r requirements-gateway.txt
```

先检查项目结构：

```bash
python -m scripts.validate_project
python tests/smoke_test.py
python -m compileall training registry serving evaluation benchmark tests scripts
```

通过标准：

```text
Project validation passed.
smoke tests passed
compileall 无报错
```

## 2. 本地无 GPU 演示

这一步不训练模型，不启动真实 vLLM。它用 mock backend 验证整个服务链路：

```text
mock vLLM
-> Gateway
-> task routing
-> metrics
-> benchmark
```

### 2.1 启动 mock backend

终端 1：

```bash
cd /home/cyh/llm+infra
source .venv/bin/activate

python -m serving.mock_vllm --host 0.0.0.0 --port 8001
```

### 2.2 启动 Gateway

终端 2：

```bash
cd /home/cyh/llm+infra
source .venv/bin/activate

uvicorn serving.gateway:app --host 0.0.0.0 --port 8000
```

### 2.3 测试 code 路由

```bash
curl http://localhost:8000/v1/task/chat \
  -H "Content-Type: application/json" \
  -d '{
    "task": "code",
    "messages": [{"role": "user", "content": "Write a Python function to add two numbers."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

期望看到：

```text
"adapter": "code-lora"
"model": "code-lora"
```

### 2.4 测试 math 路由

```bash
curl http://localhost:8000/v1/task/chat \
  -H "Content-Type: application/json" \
  -d '{
    "task": "math",
    "messages": [{"role": "user", "content": "What is 12 times 7?"}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

期望看到：

```text
"adapter": "math-lora"
"model": "math-lora"
```

### 2.5 测试 general 路由

```bash
curl http://localhost:8000/v1/task/chat \
  -H "Content-Type: application/json" \
  -d '{
    "task": "general",
    "messages": [{"role": "user", "content": "Explain LoRA in two sentences."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

期望看到：

```text
"adapter": "base"
"model": "Qwen/Qwen2.5-1.5B-Instruct"
```

### 2.6 查看 metrics

```bash
curl http://localhost:8000/metrics
```

应该能看到：

```text
llm_request_total
llm_request_latency_seconds
llm_request_errors_total
llm_tokens_generated_total
llm_adapter_requests_total
```

### 2.7 跑 mock benchmark

```bash
python -m benchmark.loadgen \
  --config configs/benchmark.yaml \
  --url http://localhost:8000/v1/task/chat \
  --concurrency 2 \
  --duration 10 \
  --output benchmark/results/mock_c2.jsonl

python -m benchmark.analyze_results benchmark/results/mock_c2.jsonl \
  --output benchmark/results/mock_c2_summary.json
```

产物：

```text
benchmark/results/mock_c2.jsonl
benchmark/results/mock_c2_summary.json
```

注意：mock benchmark 只证明工程链路能跑，不代表真实模型性能，不能当作最终简历结果。

## 3. 构建训练数据

### 3.1 小样本数据 smoke

如果服务器可以访问 Hugging Face：

```bash
python -m training.build_dataset \
  --task all \
  --code-limit 100 \
  --math-limit 100 \
  --output-dir data/processed_smoke
```

如果服务器不能访问 Hugging Face，先用仓库自带 tiny sample 跑离线 smoke：

```bash
python -m training.build_dataset \
  --task all \
  --code-local-file data/examples/codealpaca_sample.jsonl \
  --math-local-train-file data/examples/gsm8k_train_sample.jsonl \
  --math-local-test-file data/examples/gsm8k_test_sample.jsonl \
  --output-dir data/processed_smoke
```

检查产物：

```bash
ls data/processed_smoke
```

应该有：

```text
code_train.jsonl
code_valid.jsonl
code_test.jsonl
math_train.jsonl
math_valid.jsonl
math_test.jsonl
```

### 3.2 完整数据构建

如果服务器能访问 Hugging Face：

```bash
python -m training.build_dataset \
  --task all \
  --output-dir data/processed
```

如果服务器不能访问 Hugging Face，在本地或其他能访问网络的机器下载/导出 CodeAlpaca 和 GSM8K 为 JSONL 后传到服务器，然后运行：

```bash
python -m training.build_dataset \
  --task all \
  --code-local-file /path/to/codealpaca_train.jsonl \
  --math-local-train-file /path/to/gsm8k_train.jsonl \
  --math-local-test-file /path/to/gsm8k_test.jsonl \
  --output-dir data/processed
```

本地文件字段要求：

```text
CodeAlpaca JSONL: instruction, input, output
GSM8K JSONL: question, answer
```

产物：

```text
data/processed/code_train.jsonl
data/processed/code_valid.jsonl
data/processed/code_test.jsonl
data/processed/math_train.jsonl
data/processed/math_valid.jsonl
data/processed/math_test.jsonl
```

数据来源：

- Code SFT: `sahil2801/CodeAlpaca-20k`
- Math SFT/eval: `openai/gsm8k`
- Code eval: `google-research-datasets/mbpp`

## 4. 训练 QLoRA Adapter

训练前确认：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

期望：

```text
True
```

V1 的 QLoRA 训练按单卡设计。如果服务器有多张 GPU，训练命令前面要加：

```bash
CUDA_VISIBLE_DEVICES=0
```

否则 Transformers 可能自动启用 `DataParallel`，导致 4-bit QLoRA 模型训练报错。

### 4.1 code-lora 训练 smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train_qlora \
  --config configs/code_lora.yaml \
  --max-train-samples 10 \
  --max-eval-samples 5 \
  --max-steps 1
```

检查：

```bash
ls outputs/code-lora
```

至少应该有：

```text
adapter_config.json
adapter_model.safetensors
train_log.json
```

### 4.2 math-lora 训练 smoke

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train_qlora \
  --config configs/math_lora.yaml \
  --max-train-samples 10 \
  --max-eval-samples 5 \
  --max-steps 1
```

检查：

```bash
ls outputs/math-lora
```

### 4.3 完整训练 code-lora

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train_qlora \
  --config configs/code_lora.yaml
```

### 4.4 完整训练 math-lora

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train_qlora \
  --config configs/math_lora.yaml
```

### 4.5 训练后项目检查

```bash
python -m scripts.validate_project --require-outputs
```

通过标准：

```text
Project validation passed.
```

### 4.6 训练报告要记录

把下面信息填到 `report/training_report.md`：

- base model
- 数据集名称
- 样本数
- LoRA rank / alpha / dropout
- epoch
- train loss
- eval loss
- 训练耗时
- 峰值显存
- 失败问题或环境问题

## 5. Base vs LoRA 质量评测

### 5.1 跑完整评测

```bash
python -m evaluation.eval_base_vs_lora \
  --config configs/eval.yaml \
  --load-in-4bit
```

这会跑四组：

```text
GSM8K base
GSM8K math-lora
MBPP base
MBPP code-lora
```

默认规模：

```text
GSM8K: 200 samples
MBPP: 50 samples
```

### 5.2 单独跑 math eval

```bash
python -m evaluation.eval_math \
  --config configs/eval.yaml \
  --model base \
  --limit 200 \
  --load-in-4bit

python -m evaluation.eval_math \
  --config configs/eval.yaml \
  --model math-lora \
  --adapter-path outputs/math-lora \
  --limit 200 \
  --load-in-4bit
```

### 5.3 单独跑 code eval

```bash
python -m evaluation.eval_mbpp \
  --config configs/eval.yaml \
  --model base \
  --limit 50 \
  --load-in-4bit

python -m evaluation.eval_mbpp \
  --config configs/eval.yaml \
  --model code-lora \
  --adapter-path outputs/code-lora \
  --limit 50 \
  --load-in-4bit
```

### 5.4 评测产物

```text
report/results/gsm8k_eval_base.jsonl
report/results/gsm8k_eval_math-lora.jsonl
report/results/mbpp_eval_base.jsonl
report/results/mbpp_eval_code-lora.jsonl
report/results/eval_summary.json
```

### 5.5 评测报告要记录

填到 `report/eval_report.md`：

| Task | Model | Samples | Score |
| --- | --- | ---: | ---: |
| GSM8K | base | 200 | 实测 |
| GSM8K | math-lora | 200 | 实测 |
| MBPP | base | 50 | 实测 |
| MBPP | code-lora | 50 | 实测 |

还要挑失败案例：

- math-lora 错在哪里
- code-lora 生成代码为什么没过测试
- LoRA 是否只提升目标任务
- 如果没提升，分析数据量、训练步数、base model 已经较强等原因

## 6. 启动真实 vLLM Multi-LoRA Serving

### 6.1 启动 vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --enable-lora \
  --lora-modules code-lora=outputs/code-lora math-lora=outputs/math-lora
```

### 6.2 测试 vLLM 是否可用

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "code-lora",
    "messages": [{"role": "user", "content": "Write a Python function to add two numbers."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

再测 math-lora：

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "math-lora",
    "messages": [{"role": "user", "content": "What is 12 times 7?"}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

再测 base：

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain LoRA in two sentences."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

## 7. 启动 Gateway

另开终端：

```bash
cd /home/cyh/llm+infra
source .venv/bin/activate

uvicorn serving.gateway:app --host 0.0.0.0 --port 8000
```

测试 Gateway：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/adapters
curl http://localhost:8000/metrics
```

再测试 task routing：

```bash
curl http://localhost:8000/v1/task/chat \
  -H "Content-Type: application/json" \
  -d '{
    "task": "code",
    "messages": [{"role": "user", "content": "Write a Python function to reverse a string."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

确认返回：

```text
"adapter": "code-lora"
"model": "code-lora"
```

## 8. 真实 Benchmark 实验

Benchmark workload 默认在 `configs/benchmark.yaml`：

```text
50% code
40% math
10% general
```

### 8.1 concurrency=1

```bash
python -m benchmark.loadgen \
  --config configs/benchmark.yaml \
  --url http://localhost:8000/v1/task/chat \
  --concurrency 1 \
  --duration 60 \
  --output benchmark/results/real_c1.jsonl

python -m benchmark.analyze_results benchmark/results/real_c1.jsonl \
  --output benchmark/results/real_c1_summary.json
```

### 8.2 concurrency=4

```bash
python -m benchmark.loadgen \
  --config configs/benchmark.yaml \
  --url http://localhost:8000/v1/task/chat \
  --concurrency 4 \
  --duration 60 \
  --output benchmark/results/real_c4.jsonl

python -m benchmark.analyze_results benchmark/results/real_c4.jsonl \
  --output benchmark/results/real_c4_summary.json
```

### 8.3 concurrency=8

```bash
python -m benchmark.loadgen \
  --config configs/benchmark.yaml \
  --url http://localhost:8000/v1/task/chat \
  --concurrency 8 \
  --duration 60 \
  --output benchmark/results/real_c8.jsonl

python -m benchmark.analyze_results benchmark/results/real_c8.jsonl \
  --output benchmark/results/real_c8_summary.json
```

### 8.4 Benchmark 报告要记录

填到 `report/benchmark_report.md`：

| Concurrency | RPS | Tokens/s | p50 ms | p95 ms | p99 ms | Error Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| 4 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| 8 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

还要记录 adapter 分布：

| Adapter | Requests | Share |
| --- | ---: | ---: |
| code-lora | 实测 | 实测 |
| math-lora | 实测 | 实测 |
| base | 实测 | 实测 |

## 9. Prometheus / 监控实验

Gateway 启动后，直接访问：

```bash
curl http://localhost:8000/metrics
```

如果用 Docker Compose mock demo：

```bash
docker compose -f deploy/docker-compose.yaml up
```

Prometheus 地址：

```text
http://localhost:9090
```

Prometheus 里可以查：

```text
llm_request_total
llm_request_latency_seconds_count
llm_request_latency_seconds_sum
llm_request_errors_total
llm_tokens_generated_total
llm_adapter_requests_total
```

报告里建议截图：

- Prometheus targets 页面
- `llm_request_total` 查询结果
- `llm_adapter_requests_total` 查询结果

## 10. 最终报告和 README 要补什么

### 10.1 README

补真实结果表：

```text
GSM8K base vs math-lora
MBPP base vs code-lora
benchmark concurrency=1/4/8
```

### 10.2 report/training_report.md

补训练配置和训练结果。

### 10.3 report/eval_report.md

补质量评测结果和失败案例。

### 10.4 report/benchmark_report.md

补性能压测结果和 adapter 分布。

### 10.5 report/architecture.md

如果真实运行中改了模型名、端口或 serving 方式，要同步更新架构说明。

## 11. 最终验收标准

代码验收：

```bash
python -m scripts.validate_project
python tests/smoke_test.py
python -m compileall training registry serving evaluation benchmark tests scripts
```

训练验收：

```text
outputs/code-lora/adapter_config.json 存在
outputs/math-lora/adapter_config.json 存在
outputs/code-lora/train_log.json 存在
outputs/math-lora/train_log.json 存在
```

评测验收：

```text
report/results/eval_summary.json 存在
report/eval_report.md 有真实分数
```

Serving 验收：

```text
vLLM 能响应 code-lora
vLLM 能响应 math-lora
Gateway /v1/task/chat 能按 task 路由
Gateway /metrics 有指标
```

Benchmark 验收：

```text
benchmark/results/real_c1_summary.json 存在
benchmark/results/real_c4_summary.json 存在
benchmark/results/real_c8_summary.json 存在
report/benchmark_report.md 有真实 p95/p99/tokens/sec
```

简历验收：

```text
README 有架构图
README 有真实 eval 表
README 有真实 benchmark 表
报告里有失败案例和 trade-off 分析
```

## 12. 推荐执行顺序

最稳顺序：

```text
1. python -m scripts.validate_project
2. python tests/smoke_test.py
3. 本地 mock backend + Gateway + benchmark
4. build_dataset 小样本 smoke
5. build_dataset 完整数据
6. code-lora 训练 smoke
7. math-lora 训练 smoke
8. code-lora 完整训练
9. math-lora 完整训练
10. base vs LoRA eval
11. 启动真实 vLLM
12. 启动 Gateway
13. concurrency=1/4/8 benchmark
14. Prometheus metrics 截图
15. 填 README 和 report
```

这 15 步做完，TaskLoRA-Serve V1 就是完整项目。
