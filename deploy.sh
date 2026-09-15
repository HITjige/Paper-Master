export CUDA_VISIBLE_DEVICES=3

# Disable Qwen3.5 thinking by default so tool-result continuations always
# produce visible content or a tool call. Individual requests can opt back in
# with chat_template_kwargs.enable_thinking=true when reasoning is required.
python -m vllm.entrypoints.openai.api_server \
  --model /data1/project/models/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8010 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enforce-eager \
  # --default-chat-template-kwargs '{"enable_thinking": false}' \
