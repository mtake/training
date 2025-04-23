# %% [markdown]
# Below is an example recipe for training a model with reasoning traces, to become a "thinking model". In this example, we utilize Microsoft's `Phi-4-mini-instruct` and NVIDIA's Nemotron Post-Training Dataset for reasoning/non-reasoning traces.

# %% [markdown]
# # SFT: Warm Start for Reasoning
# 
# The first step in this process is to introduce Phi-4-mini-instruct to the structure and style of thought, in multiple contexts. We also want to keep the Nemotron method of reasoning-toggle-via-system-prompt, so we need the model to see examples of reasoning and non-reasoning responses, with detailed thinking on and detailed thinking off as corresponding system prompts.
# 

# %% [markdown]
# ## Data Preparation

# %%
data_prep_num_proc = 8

# %% [markdown]
# To accomplish this, we use the open source Nemotron Post-Training Dataset, but it cannot be used as-is. The dataset is specific to Llama, and includes 15 million samples (most of which were unused in Nemotron training), so we will convert and filter the dataset to a more digestible messages-format set of samples, usable by any model. We start by loading the dataset via Huggingface Datasets:

# %%
from datasets import load_dataset, concatenate_datasets

print("Start loading dataset")

dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1")

print("Finished loading dataset")

# %% [markdown]
# We then take each category in the SFT data subset, and generalize the samples used in Nemotron training:

# %%
def generalize_sample(sample):
    user = sample["input"].split("user<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0]
    assistant = sample["output"].replace("<|eot_id|>", '')
    message_list = [
        {"role": "system", "content": f"detailed thinking {sample['reasoning']}"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return {"messages": message_list}

generic_samples_datasets = []
for split in dataset.keys():
    print(f"Processing {split} samples")
    new_split = dataset[split].filter(lambda sample: sample["used_in_training"] == 'yes', num_proc=data_prep_num_proc)
    print(f"Adding {len(new_split)} samples")
    new_samples = new_split.map(generalize_sample, remove_columns=list(new_split[0].keys()), num_proc=data_prep_num_proc)
    generic_samples_datasets.append(new_samples)
    print("Samples added\n")

# %% [markdown]
# Once we’ve got all of our reduced, generalized samples, we can re-combine them into a single dataset and save as a jsonl:

# %%
print("Writing generic messages-format data")
generic_samples = concatenate_datasets(generic_samples_datasets)
print(generic_samples)
generic_samples.to_json("nemotron.jsonl", lines=True, orient="records", num_proc=data_prep_num_proc)
print("Write complete!")

# %% [markdown]
# This leaves us with 1.7 million samples of math, science, code, chat, and safety. This includes examples with and without detailed reasoning. With this file, we are ready to start SFT.

# %% [markdown]
# ## Fine-Tuning

# %%
fine_tune_nproc_per_node = 8
fine_tune_nnodes = 1

# %% [markdown]
# For fine-tuning, we use the Instructlab Training library, built for optimal and efficient fine-tuning on any messages-format data. Using the python interface, we are able to launch the model training.
# 
# In this case, we ensure that we install off of main, to get the latest generic Causal LM support:

# %%
#! pip install git+https://github.com/instructlab/training.git@main

# %% [markdown]
# We start by importing the necessary pieces from the library:

# %%
from instructlab.training.config import TorchrunArgs,TrainingArgs,DistributedBackend,FSDPOptions
from instructlab.training.main_ds import run_training

# %% [markdown]
# We then define our distributed settings via TorchrunArgs. In our case, we trained on a single node with 8 H100 GPUs:

# %%
torch_args = TorchrunArgs(
	nproc_per_node=fine_tune_nproc_per_node,
	nnodes=fine_tune_nnodes,
 	node_rank=0,
	rdzv_id=123,
 	rdzv_endpoint="0.0.0.0:8888",
)

# %% [markdown]
# We then set our model and data paths, checkpoint output path, and hyperparameters via the TrainingArgs object:

# %%
train_args = TrainingArgs(
	model_path="microsoft/Phi-4-mini-instruct",
	data_path="nemotron.jsonl",
	ckpt_output_dir="experiments/training_output",
	data_output_dir="data/processed-data",                    # processed data ids/labels/masks
	max_seq_len=20000,
	max_batch_len=30000,                                      # max tokens per gpu
	num_epochs=3, 
	effective_batch_size=256,                                 # target batch size per model update
	learning_rate=2e-5,
	warmup_steps=25,
    save_samples=0,                                           # save ckpt after num of samples seen (0=off)
    checkpoint_at_epoch = True,                               # save ckpt after every epoch
    accelerate_full_state_at_epoch = False,                   # save full-state for resuming
    process_data=True,                                        # can set to false if data processed before
	distributed_backend=DistributedBackend.FSDP,
	fsdp_options=FSDPOptions(cpu_offload_params=False),
)

# %% [markdown]
# Finally, we kick off SFT via the run_training function:

# %%
run_training(torch_args=torch_args,train_args=train_args)

# %% [markdown]
# Upon completion, we have n (n=num_epochs) Huggingface-Format checkpoints in `experiments/training_output/hf_format`. The full run logs and metrics will also be recorded in `experiments/training_output`. Running the final training as a python script rather than in a notebook may help with progress bar writing to stdout.


