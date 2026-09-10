export CUDA_VISIBLE_DEVICES=3

# Keep Qwen3.5 thinking available for agent requests. Short structured calls
# such as chunk-metadata extraction disable it per request with
# chat_template_kwargs.enable_thinking=false; do not disable it globally here.
python -m vllm.entrypoints.openai.api_server \
  --model /data1/project/models/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8010 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enforce-eager
