import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "./results/llama2/final_merged_checkpoint"
    #low_cpu_mem_usage=True,
    #torch_dtype=torch.float16,
    #load_in_4bit=True,
) 
tokenizer = AutoTokenizer.from_pretrained("./results/llama2/final_merged_checkpoint")

prompt = f"""How to play chess?
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids#.cuda()
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)

print(f"Generated output:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")