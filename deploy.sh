export CUDA_VISIBLE_DEVICES=1

python -m vllm.entrypoints.openai.api_server \
    --model /data1/project/models/Qwen3-4B-Instruct \
    --tensor_parallel_size 1 \
    --host 0.0.0.0 \
    --port 8010 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \