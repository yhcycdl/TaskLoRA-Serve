# TaskLoRA-Serve 代码阅读指南

这份文档帮你快速读懂这个项目的代码。建议你按“数据流”读，而不是按文件名从上到下硬啃。

## 1. 这个项目到底在做什么

TaskLoRA-Serve V1 是一个小而完整的 AI Infra 项目：

```text
公开数据集
-> 转成 chat-format SFT 数据
-> 训练两个 QLoRA Adapter
-> 评测 base vs LoRA
-> vLLM 加载 base model + adapters
-> FastAPI Gateway 按 task 选择 adapter
-> 压测脚本统计延迟、吞吐、adapter 分布
-> Prometheus 暴露服务指标
```

它不是普通聊天机器人。项目重点是证明你理解大模型工程链路里的几个关键点：

- 数据如何变成训练样本
- LoRA/QLoRA adapter 如何训练和保存
- base model 和 task adapter 如何对比评测
- vLLM OpenAI-compatible API 如何作为推理后端
- Gateway 如何隐藏后端细节并做任务路由
- benchmark 和 metrics 如何证明系统表现

## 2. 推荐阅读顺序

### 第一步：先读配置

先看这些文件：

- `configs/model.yaml`
- `configs/code_lora.yaml`
- `configs/math_lora.yaml`
- `configs/serving.yaml`
- `registry/adapters.yaml`

你要先搞清楚：

- base model 固定是 `Qwen/Qwen2.5-1.5B-Instruct`
- adapter 有两个：`code-lora` 和 `math-lora`
- `task=code` 路由到 `code-lora`
- `task=math` 路由到 `math-lora`
- `task=general` 路由到 base model

这一步读明白后，后面所有脚本都只是围绕这些配置工作。

### 第二步：读数据构建

核心文件：

- `training/build_dataset.py`
- `training/train_utils.py`

`build_dataset.py` 做三件事：

1. 下载 Hugging Face 数据集。
2. 把 CodeAlpaca 和 GSM8K 转成统一的 `messages` 格式。
3. 输出 `jsonl` 文件到 `data/processed/`。

Code 样本结构大概是：

```json
{
  "task": "code",
  "messages": [
    {"role": "system", "content": "...programming assistant..."},
    {"role": "user", "content": "Write a function..."},
    {"role": "assistant", "content": "def ..."}
  ]
}
```

Math 样本结构类似，只是 system prompt 和任务来源换成 GSM8K。

### 第三步：读训练脚本

核心文件：

- `training/train_qlora.py`

这份脚本负责：

- 读取 `configs/code_lora.yaml` 或 `configs/math_lora.yaml`
- 加载 tokenizer 和 base model
- 用 bitsandbytes 做 4-bit QLoRA
- 用 PEFT 包一层 LoRA adapter
- 把 chat messages 渲染成模型训练文本
- 用 Hugging Face `Trainer` 训练
- 保存 adapter 到 `outputs/code-lora/` 或 `outputs/math-lora/`

读这份代码时重点看：

- `BitsAndBytesConfig`：为什么能低显存训练
- `LoraConfig`：rank、alpha、target modules 怎么配置
- `tokenize_batch`：messages 如何变成 token
- `trainer.save_model`：adapter 最终保存在哪里

### 第四步：读评测

核心文件：

- `evaluation/eval_math.py`
- `evaluation/eval_mbpp.py`
- `evaluation/eval_base_vs_lora.py`
- `evaluation/eval_utils.py`

数学评测用 GSM8K：

- 生成答案
- 从模型输出里抽取最终数字
- 和标准答案做 exact match

代码评测用 MBPP：

- 让模型生成 Python 代码
- 抽取代码块
- 执行 MBPP 的 `test_list`
- 统计 pass@1

这个设计比 code-review/bugfix 更容易落地，因为结果更客观，不太依赖人工主观评分。

### 第五步：读 Registry 和 Router

核心文件：

- `registry/model_registry.py`
- `serving/router.py`

`ModelRegistry` 读取 `registry/adapters.yaml`，负责告诉系统：

- 有哪些 adapter
- adapter 对应什么 task
- adapter 在 vLLM 里的 serving name 是什么
- adapter 文件路径在哪里

`TaskRouter` 负责把业务 task 转成后端 model name：

```text
code -> code-lora
math -> math-lora
general -> Qwen/Qwen2.5-1.5B-Instruct
```

这里是项目 AI Infra 味道的入口：客户端不需要知道 LoRA 名字，只需要传 `task`。

### 第六步：读 Gateway

核心文件：

- `serving/request_schema.py`
- `serving/gateway.py`
- `serving/mock_vllm.py`
- `serving/vllm_backend.py`
- `observability/metrics.py`

`request_schema.py` 定义对外 API：

```json
{
  "task": "math",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "max_tokens": 512,
  "temperature": 0.2
}
```

`gateway.py` 是服务入口：

```text
request
-> validate
-> router.route(task)
-> vLLMBackend.chat_completions(...)
-> record metrics
-> return response
```

`vllm_backend.py` 只负责调用 vLLM 的 `/v1/chat/completions`。

`mock_vllm.py` 是本地无 GPU 演示用的 OpenAI-compatible 假后端。它不跑模型，但能验证 Gateway 路由、metrics 和 benchmark 链路。

`metrics.py` 负责暴露：

- `llm_request_total`
- `llm_request_latency_seconds`
- `llm_request_errors_total`
- `llm_tokens_generated_total`
- `llm_adapter_requests_total`

### 第七步：读 Benchmark

核心文件：

- `benchmark/loadgen.py`
- `benchmark/analyze_results.py`
- `configs/benchmark.yaml`

`loadgen.py` 会按比例生成混合请求：

- 50% code
- 40% math
- 10% general

每个请求都会记录：

- task
- adapter
- latency
- tokens
- status
- error

`analyze_results.py` 汇总：

- requests/sec
- tokens/sec
- p50/p95/p99 latency
- error rate
- adapter distribution

这是简历里最该展示的部分，因为它能证明你不是只“接了个模型”，而是真的测了服务表现。

## 3. 怎么把项目跑起来

### 只读代码和跑 smoke test

当前机器没有 GPU 也能跑：

```bash
python tests/smoke_test.py
python -m compileall training registry serving evaluation benchmark tests scripts
python -m scripts.validate_project
```

如果想演示完整服务链路但没有 GPU，看 `RUNBOOK.md` 里的 mock backend 流程。

### 构建小数据集

需要安装 `datasets`：

```bash
python -m training.build_dataset \
  --task all \
  --code-limit 100 \
  --math-limit 100 \
  --output-dir data/processed_smoke
```

### 训练 smoke test

需要 Python 3.10/3.11、PyTorch、Transformers、PEFT、TRL、bitsandbytes：

```bash
python -m training.train_qlora \
  --config configs/code_lora.yaml \
  --max-train-samples 10 \
  --max-eval-samples 5 \
  --max-steps 1
```

### 启动 vLLM

训练完 adapter 后：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --enable-lora \
  --lora-modules code-lora=outputs/code-lora math-lora=outputs/math-lora
```

### 启动 Gateway

```bash
uvicorn serving.gateway:app --host 0.0.0.0 --port 8000
```

### 压测

```bash
python -m benchmark.loadgen \
  --config configs/benchmark.yaml \
  --url http://localhost:8000/v1/task/chat \
  --concurrency 4 \
  --duration 60 \
  --output benchmark/results/v1_c4.jsonl
```

## 4. 面试时怎么讲这套代码

可以按这个顺序讲：

1. 我没有做普通 RAG，而是做了一个多任务 LoRA serving 系统。
2. 我选 CodeAlpaca 和 GSM8K，是因为数据现成、任务明确、评测更客观。
3. 我用 QLoRA 训练两个 adapter，分别服务 code 和 math 请求。
4. 我用 vLLM 做 OpenAI-compatible serving，Gateway 根据 task 选择 adapter。
5. 我用 benchmark 统计 p95/p99 latency、tokens/sec、error rate 和 adapter 分布。
6. 我用 Prometheus metrics 暴露服务指标，让项目更像真实生产系统。

最重要的点不是“我用了很多库”，而是：

```text
训练、评测、推理、路由、压测、监控是一条完整闭环。
```

## 5. 目前还可以继续完善的地方

优先级从高到低：

1. 在 Python 3.10/3.11 GPU 环境跑通 10-sample training smoke。
2. 跑完整 CodeAlpaca/GSM8K 训练，填 `report/training_report.md`。
3. 跑 base vs LoRA eval，填 `report/eval_report.md`。
4. 启动 vLLM + Gateway，跑 concurrency=1/4/8 benchmark，填 `report/benchmark_report.md`。
5. 给 README 加真实结果表和 Prometheus 截图。
6. V1.5 再加 Grafana dashboard。
7. V2 再考虑 prefix-cache-aware routing、SGLang 对比、Kubernetes。
