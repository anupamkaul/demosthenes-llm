'''
Stylize downloaded dataset into common style prompts used by LLMs

(see images: prompt_styles.png, we go with Alpaca style here - more popular
and helped define original approach to fine-tuning) 
'''

def format_input(entry):

    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request. "
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text



