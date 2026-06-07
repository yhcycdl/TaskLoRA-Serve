# TaskLoRA-Serve 运行手册

这份手册说明怎么从“本地无 GPU 演示”走到“真实 GPU 训练和 vLLM Serving”。

## 1. 本地无 GPU 演示链路

这条链路不训练模型，也不启动真实 vLLM。它用 `serving/mock_vllm.py` 模拟 OpenAI-compatible 后端，用来验证：

- Gateway 可以启动
- task 路由可以工作
- `/metrics` 可以暴露指标
- benchmark 可以打通

安装轻量依赖：

```bash
pip install -r requirements-gateway.txt
```

启动 mock vLLM：

```bash
python -m serving.mock_vllm --host 0.0.0.0 --port 8001
```

另开一个终端启动 Gateway：

```bash
uvicorn serving.gateway:app --host 0.0.0.0 --port 8000
```

测试路由：

```bash
curl http://localhost:8000/v1/task/chat \
  -H "Content-Type: application/json" \
  -d '{
    "task": "code",
    "messages": [{"role": "user", "content": "Write a Python add function."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

你应该看到返回里的：

```text
"adapter": "code-lora"
"model": "code-lora"
```

再测 math：

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

你应该看到：

```text
"adapter": "math-lora"
"model": "math-lora"
```

查看 metrics：

```bash
curl http://localhost:8000/metrics
```

运行短压测：

```bash
python -m benchmark.loadgen \
  --config configs/benchmark.yaml \
  --url http://localhost:8000/v1/task/chat \
  --concurrency 2 \
  --duration 10 \
  --output benchmark/results/mock_c2.jsonl

python -m benchmark.analyze_results benchmark/results/mock_c2.jsonl
```

这一步能产生 p50/p95/p99 latency、requests/sec、tokens/sec、adapter 分布。注意：mock 后端结果只能证明工程链路，不代表真实模型性能。

也可以用 Docker Compose 一键启动 mock backend、Gateway 和 Prometheus：

```bash
docker compose -f deploy/docker-compose.yaml up
```

服务地址：

- Gateway: `http://localhost:8000`
- Mock backend: `http://localhost:8001`
- Prometheus: `http://localhost:9090`

## 2. 项目结构检查

```bash
python -m scripts.validate_project
python tests/smoke_test.py
python -m compileall training registry serving evaluation benchmark tests scripts
```

训练完成后可以加更严格检查：

```bash
python -m scripts.validate_project --require-outputs
```

## 3. 数据构建

推荐先小样本 smoke：

```bash
python -m training.build_dataset \
  --task all \
  --code-limit 100 \
  --math-limit 100 \
  --output-dir data/processed_smoke
```

完整 V1 数据：

```bash
python -m training.build_dataset --task all --output-dir data/processed
```

## 4. 训练

训练建议使用 Python 3.10/3.11 + CUDA 对应版本 PyTorch。建议先单独安装 PyTorch，再安装项目依赖：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

先跑 smoke：

```bash
python -m training.train_qlora \
  --config configs/code_lora.yaml \
  --max-train-samples 10 \
  --max-eval-samples 5 \
  --max-steps 1
```

完整训练：

```bash
python -m training.train_qlora --config configs/code_lora.yaml
python -m training.train_qlora --config configs/math_lora.yaml
```

训练结束后检查：

```bash
ls outputs/code-lora/adapter_config.json
ls outputs/math-lora/adapter_config.json
```

## 5. 真实 vLLM Serving

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --enable-lora \
  --lora-modules code-lora=outputs/code-lora math-lora=outputs/math-lora
```

启动 Gateway：

```bash
uvicorn serving.gateway:app --host 0.0.0.0 --port 8000
```

## 6. 评测和报告

```bash
python -m evaluation.eval_base_vs_lora --config configs/eval.yaml --load-in-4bit
```

把结果填进：

- `report/training_report.md`
- `report/eval_report.md`
- `report/benchmark_report.md`

## 7. 常见问题

### Gateway 返回 502

通常是 vLLM/mock backend 没启动，或者 `configs/serving.yaml` 里的 `vllm_base_url` 不对。

### general 请求失败

确认 `configs/serving.yaml` 里的 `base_model_serving_name` 和 vLLM 启动后的模型名一致。默认是 `Qwen/Qwen2.5-1.5B-Instruct`。

### 训练依赖装不上

不要用 Python 3.13 训练。改用 Python 3.10 或 3.11。

### mock benchmark 数据很好看

mock benchmark 只证明 Gateway 和压测系统能工作，不代表真实模型吞吐。简历和报告里必须标明真实 vLLM benchmark 结果。
