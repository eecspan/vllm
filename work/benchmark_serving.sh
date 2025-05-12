vllm serve /workspace/models/Qwen1.5-32B-Chat/ -tp=4 --swap-space=16 --disable-log-requests

python benchmarks/benchmark_serving.py --model /workspace/models/Qwen1.5-32B-Chat/ --dataset-name random --backend vllm --num-prompts 12 --random-input-len 128 --random-output-len 1024

export VLLM_ENABLE_L2CACHE_PRELOAD=1