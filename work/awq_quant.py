from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/workspace/models/Qwen1.5-32B-Chat'
quant_path = '/workspace/models/Qwen1.5-32B-Chat-w4a16'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')

