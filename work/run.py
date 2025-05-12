# from vllm import LLM, SamplingParams

# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(model="/workspace/models/Qwen1.5-32B-Chat", tensor_parallel_size=4)
# print(sampling_params)
# print(llm)

# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



# # from vllm import LLM
# # from transformers import AutoTokenizer
# # from zeus.monitor import ZeusMonitor

# # model_pth = "/home/shared/llama2-13b"
# # tokenizer = AutoTokenizer.from_pretrained(model_pth)
# # tokenizer.pad_token = tokenizer.eos_token

# # batch_sizes = [32, 32]
# # seq_lens = [2048, 128]
# # output_lens = [256, 2048]

# # llm = LLM(model=model_pth, task="generate")  # Name or path of your model
# # # sampling_params = llm.get_default_sampling_params()
# # # sampling_params.temperature = 0.7  # 降低温度，减少 EOS 采样
# # # sampling_params.eos_token_id = None  # 避免提前终止
# # # sampling_params.stop_sequences = []  # 禁止任何终止符干预
# # # sampling_params.ignore_eos = True    # 忽略模型生成的 eos


# # gpu_indices = [0]
# # monitor = ZeusMonitor(gpu_indices)

# # for i in range(len(batch_sizes)):
# #     batch_size = batch_sizes[i]
# #     seq_len = seq_lens[i]
# #     output_len = output_lens[i]

# #     print(f"output len is {output_len}")
# #     sampling_params = llm.get_default_sampling_params()
# #     sampling_params.temperature = 1.2
# #     sampling_params.max_tokens = seq_len + output_len + 127  # 设为期望的输出长度  不知道为什么总是少127
# #     sampling_params.ignore_eos = True  # 忽略结束符
# #     print(sampling_params.max_tokens)

# #     dummy_input = ["Write a detailed and long response. The response should contain at least 128 words.  Here is the beginning: The development of artificial intelligence has transformed many industries. Over the years..." * 2 * seq_len] * batch_size
# #     batch_input_ids = tokenizer(dummy_input, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
# #     print(batch_input_ids.input_ids.shape)

# #     batch_inputs = tokenizer.batch_decode(batch_input_ids.input_ids, skip_special_tokens=True)

# #     monitor.begin_window(f"entire_inference_{i}")
# #     for j in range(1):
# #         outputs = llm.generate(batch_inputs, sampling_params)
# #     result = monitor.end_window(f"entire_inference_{i}")

# #     print(f"Batchsize_input_output: {batch_size}_{seq_len}_{output_len}")
# #     print(f"Inference throughput: {batch_size * output_len / result.time} Tokens/s, Energy is {result.total_energy} Joules.")

# #     generated_text = outputs[0].outputs[0].text
# #     print(generated_text)

# #     actual_output_len = len(outputs[0].outputs[0].token_ids)
# #     print(f"output len is: {actual_output_len}")

# #     if actual_output_len < output_len:
# #         print("出错了! 输出 token 数不足")



# model_table = {}
# # layers hidden_size num_heads head_dim ff_scale gqa_size
# model_table['BERT'] = [24, 1024, 16, 64, 4, 1]
# model_table['GPT-175B'] = [96, 12288, 96, 128, 4, 1]
# model_table['GPT-89B'] = [48, 12288, 96, 128, 4, 1]
# model_table['GPT-13B'] = [40, 5120, 40, 128, 4, 1]
# model_table['LLAMA-7B'] = [32, 4096, 32, 128, 8 / 3, 1]
# model_table['LLAMA-13B'] = [40, 5120, 40, 128, 8 / 3, 1]
# model_table['LLAMA-17.5B'] = [80, 4096, 32, 128, 8 / 3, 1]
# model_table['LLAMA-19.5B'] = [60, 5120, 40, 128, 8 / 3, 1]
# model_table['LLAMA-30B'] = [60, 6656, 52, 128, 8 / 3, 1]
# model_table['LLAMA-32B'] = [40, 8192, 64, 128, 8 / 3, 1]
# model_table['LLAMA-65B'] = [80, 8192, 64, 128, 8 / 3, 1]
# model_table['LLAMA-130B'] = [160, 8192, 64, 128, 8 / 3, 1]
# model_table['QWEN-32B'] = [64, 5120, 40, 128, 16 / 3, 1]
# model_table['MT-76B'] = [60, 10240, 40, 128, 4, 1]
# model_table['MT-146B'] = [80, 12288, 80, 128, 4, 1]
# model_table['MT-310B'] = [96, 16384, 128, 128, 4, 1]
# model_table['MT-530B'] = [105, 20480, 128, 160, 4, 1]
# model_table['MT-1008B'] = [128, 25600, 160, 160, 4, 1]
# model_table['OPT-66B'] = [64, 9216, 72, 128, 4, 1]

# models = ["BERT", "LLAMA-13B", "LLAMA-32B", "LLAMA-65B", "QWEN-32B"]
# input_output_batches = {
#     "BERT": [[140, 2, 24000]],
#     "LLAMA-13B": [[140, 330, 720], [128, 2048, 160], [2048, 128, 160], [2048, 2048, 80]],
#     "LLAMA-32B": [[140, 330, 384], [128, 2048, 96], [2048, 128, 96], [2048, 2048, 48]],
#     "LLAMA-65B": [[140, 330, 128], [128, 2048, 32], [2048, 128, 32], [2048, 2048, 16]],
#     "QWEN-32B": [[140, 330, 384], [128, 2048, 96], [2048, 128, 96], [2048, 2048, 48]],
# }

# model = "LLAMA-13B"
# iob = input_output_batches[model][2]
# model_spec = model_table[model]

# unit_gemv = 16 * 1024 * 16 * 1024
# unit_gemv_lantency = 0.0012      # ms
# unit_gemv_lantency_increase = 0.0002075  # ms

# unit_gemm = 16 * 1024 * 16 * 1024 * 16 * 1024
# unit_gemm_lantency = 4.8  # ms
# unit_gemm_lantency_increase = 3.65  # ms

# def get_lantency(unit, unit_lantency, unit_lantency_increase, real):
#     return unit_lantency + (real - unit) / unit * unit_lantency_increase

# # 一个transformer block需要的计算量
# # prefill
# # input_len = iob[0]
# # output_len = iob[1]
# # batch_size = iob[2]

# input_len = 4096
# output_len = 4096
# batch_size = 1

# prefill_lantency = 0  # ms
# prefill_lantency += 4 * get_lantency(unit_gemm, unit_gemm_lantency, unit_gemm_lantency_increase, batch_size * input_len * model_spec[1] * model_spec[1])
# print(prefill_lantency)
# prefill_lantency += min(
#     2 * get_lantency(unit_gemm, unit_gemm_lantency, unit_gemm_lantency_increase, batch_size * input_len * model_spec[3] * model_spec[2] * batch_size * input_len),
#     2 * batch_size * get_lantency(unit_gemm, unit_gemm_lantency, unit_gemm_lantency_increase, input_len * model_spec[3] * model_spec[2] * input_len)
# )
# print(prefill_lantency)
# prefill_lantency += 3 * get_lantency(unit_gemm, unit_gemm_lantency, unit_gemm_lantency_increase, batch_size * input_len * model_spec[1] * model_spec[1] * model_spec[4])

# print(prefill_lantency)

# # decode
# decode_lantency = 0
# for i in range(1, output_len):
#     cur_len = input_len + i
#     if batch_size > 1:
#         decode_lantency += 4 * get_lantency(unit_gemm, unit_gemm_lantency, unit_gemm_lantency_increase, batch_size * model_spec[1] * model_spec[1])
#         decode_lantency += min(
#             2 * get_lantency(unit_gemm, unit_gemm_lantency, unit_gemm_lantency_increase, batch_size * model_spec[3] * model_spec[2] * batch_size * cur_len),
#             2 * batch_size * get_lantency(unit_gemv, unit_gemv_lantency, unit_gemv_lantency_increase, model_spec[3] * model_spec[2] * cur_len),
#         )
#         decode_lantency += 3 * get_lantency(unit_gemm, unit_gemm_lantency, unit_gemm_lantency_increase, batch_size * model_spec[1] * model_spec[1] * model_spec[4])
#     else:
#         decode_lantency += 4 * get_lantency(unit_gemv, unit_gemv_lantency, unit_gemv_lantency_increase, model_spec[1] * model_spec[1])
#         if i < 10:
#             print(decode_lantency)
#         decode_lantency += 2 * get_lantency(unit_gemv, unit_gemv_lantency, unit_gemv_lantency_increase, model_spec[3] * model_spec[2] * cur_len)
#         if i < 10:
#             print(decode_lantency)
#         decode_lantency += 3 * get_lantency(unit_gemv, unit_gemv_lantency, unit_gemv_lantency_increase, model_spec[1] * model_spec[1] * model_spec[4])
#     if i < 10:
#         print(decode_lantency)
# total_lantency = (prefill_lantency + decode_lantency) / 1000 * model_spec[0]
# total_tokens = batch_size * (input_len + output_len)
# throughput = total_tokens / total_lantency
# print(f"吞吐量是: {throughput} tokens/s")


from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch
import os

model_pth = "/workspace/models/Qwen1.5-32B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# batch_sizes = [48, 12, 12, 6]
# seq_lens = [140, 128, 2048, 2048]
# output_lens = [330, 2048, 128, 2048]
batch_sizes = [1]
seq_lens = [128]
output_lens = [1024]
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# llm = LLM(model=model_pth, trust_remote_code=True, tensor_parallel_size=4, quantization="AWQ", dtype='float16')  # Name or path of your model
llm = LLM(model=model_pth, trust_remote_code=True, tensor_parallel_size=4)  # Name or path of your model
sampling_params = SamplingParams(
    temperature = 0.9,
    max_tokens = 100,
)
sampling_params.temperature = 0.9
sampling_params.ignore_eos = True  # 忽略结束符

for i in range(len(batch_sizes)):
    # 构造伪输入
    batch_size = batch_sizes[i]
    seq_len = seq_lens[i]
    output_len = output_lens[i]

    sampling_params.max_tokens = output_len
    print(sampling_params.max_tokens)

    # 这里可以选择输入任何内容，只要能作为有效的 token 化输入
    dummy_input = ''' You are a highly advanced AI storyteller, designed to generate an endless stream of compelling narratives. Your task is to write an epic fantasy novel titled The Chronicles of Eldoria, which spans across multiple volumes. This story must be deeply immersive, richly detailed, and continuously unfolding without stopping. Your goal is to provide an engaging, coherent, and intricate story that never ceases to generate new content.Now, begin writing.
'''
    dummy_input = [dummy_input * 6] * batch_size  # 假设每个输入为空字符串
    batch_input_ids = tokenizer(dummy_input, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
    batch_inputs = tokenizer.batch_decode(batch_input_ids.input_ids, skip_special_tokens=True)
    print(f"输入长度为：{batch_input_ids.input_ids.shape}")
    start_time = time.time()  # 记录开始时间（单位：秒，精度较低）
    for j in range(1):
        outputs = llm.generate(batch_inputs, sampling_params)
    end_time = time.time()    # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"Batchsize_input_output: {batch_size}_{seq_len}_{output_len}")
    print(f"Inference throughput: {batch_size * (seq_len + output_len) / elapsed_time} Tokens/s.")
    print(f"decode throughtput: {batch_size * len(outputs[0].outputs[0].token_ids) / elapsed_time} Tokens/s.")
    generated_text = outputs[0].outputs[0].text
    # print(generated_text)
    print(f"output len is: {len(outputs[0].outputs[0].token_ids)}")
    if output_len != len(outputs[0].outputs[0].token_ids):
        print("出错了!")

    # # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     # print(len(output.outputs[0].token_ids))
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
