# %% [markdown]
# Below is an example recipe for training a model with reasoning traces, to become a "thinking model". In this example, we utilize Microsoft's `Phi-4-mini-instruct` and NVIDIA's Nemotron Post-Training Dataset for reasoning/non-reasoning traces.

# %% [markdown]
# # SFT: Warm Start for Reasoning
# 
# The first step in this process is to introduce Phi-4-mini-instruct to the structure and style of thought, in multiple contexts. We also want to keep the Nemotron method of reasoning-toggle-via-system-prompt, so we need the model to see examples of reasoning and non-reasoning responses, with detailed thinking on and detailed thinking off as corresponding system prompts.
# 

# %% [markdown]
# ## Environment Variables

# %%
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# %% [markdown]
# ## Constants

# %%
# data_name = "nemotron"
data_name = "messages_data_teigaku-genzei"

messages_data_path = f"{data_name}.jsonl"

# %% [markdown]
# ## Data Preparation

# %%
prep_data_num_proc = 8

# %%
force_prep_data = False

prep_data = not os.path.isfile(messages_data_path) or force_prep_data

# %% [markdown]
# To accomplish this, we use the open source Nemotron Post-Training Dataset, but it cannot be used as-is. The dataset is specific to Llama, and includes 15 million samples (most of which were unused in Nemotron training), so we will convert and filter the dataset to a more digestible messages-format set of samples, usable by any model. We start by loading the dataset via Huggingface Datasets:

# %%
if prep_data:
    from datasets import load_dataset, concatenate_datasets

    print("Start loading dataset", flush=True)

    # dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1")  # This redirects to "nvidia/Llama-Nemotron-Post-Training-Dataset" and the version is v1.1
    dataset = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", revision="ed905e6239c9d191e4c965a403dde07a5383b5eb")  # This is v1

    print("Finished loading dataset", flush=True)

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
if prep_data:
    for split in dataset.keys():
        print(f"Processing {split} samples", flush=True)
        new_split = dataset[split].filter(lambda sample: sample["used_in_training"] == 'yes', num_proc=prep_data_num_proc)
        print(f"Adding {len(new_split)} samples", flush=True)
        new_samples = new_split.map(generalize_sample, remove_columns=list(new_split[0].keys()), num_proc=prep_data_num_proc)
        generic_samples_datasets.append(new_samples)
        print("Samples added\n", flush=True)

# %% [markdown]
# Once we’ve got all of our reduced, generalized samples, we can re-combine them into a single dataset and save as a jsonl:

# %%
if prep_data:
    print("Writing generic messages-format data", flush=True)
    generic_samples = concatenate_datasets(generic_samples_datasets)
    print(generic_samples, flush=True)
    generic_samples.to_json(messages_data_path, lines=True, orient="records", num_proc=prep_data_num_proc)
    print("Write complete!", flush=True)

# %% [markdown]
# This leaves us with 1.7 million samples of math, science, code, chat, and safety. This includes examples with and without detailed reasoning. With this file, we are ready to start SFT.

# %% [markdown]
# ## Fine-Tuning

# %%
import torch

assert torch.cuda.is_available()
# fine_tune_nproc_per_node = 8  # original
fine_tune_nproc_per_node = torch.cuda.device_count()  # NOTE adjust to the available # of gpus
print(f"fine_tune_nproc_per_node: {fine_tune_nproc_per_node}", flush=True)

fine_tune_nnodes = 1
print(f"fine_tune_nnodes: {fine_tune_nnodes}", flush=True)

# %%
from pathlib import Path
home = Path.home()

# model_path = "microsoft/Phi-4-mini-instruct"  # OK
# model_path = f"{home}/.cache/instructlab/models/granite-3.1-8b-starter-v1"
# model_path = f"{home}/.cache/instructlab/models/granite-3.1-8b-lab-v1"  # OK
# model_path = f"{home}/.cache/instructlab/models/granite-3.1-8b-lab-v2_rev-2"  # OK
# model_path = "ibm-granite/granite-3.3-8b-base"
model_path = "ibm-granite/granite-3.3-8b-instruct"  # OK

model_name = os.path.basename(model_path)

chat_tmpl_dir = "../src/instructlab/training/chat_templates"
if "granite" in model_name:
    chat_tmpl_path = f"{chat_tmpl_dir}/ibm_generic_tmpl.py"
else:
    chat_tmpl_path = None

ckpt_output_dir = f"experiments/training_output-{model_name}-{data_name}"
processed_data_dir = f"data/processed-data-{model_name}-{data_name}"

num_epochs = 3  # original
# num_epochs = 1  # NOTE time saver

# %%
force_process_data = False

process_data = not os.path.isfile(f"{processed_data_dir}/data.jsonl") or force_process_data

# %% [markdown]
# For fine-tuning, we use the Instructlab Training library, built for optimal and efficient fine-tuning on any messages-format data. Using the python interface, we are able to launch the model training.
# 
# In this case, we ensure that we install off of main, to get the latest generic Causal LM support:

# %%
# %%capture
# %pip install git+https://github.com/instructlab/training.git@main

# %% [markdown]
# We start by importing the necessary pieces from the library:

# %%
from instructlab.training.config import TorchrunArgs, TrainingArgs, DistributedBackend, FSDPOptions
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
	model_path=model_path,
	chat_tmpl_path=chat_tmpl_path,
	data_path=messages_data_path,
	ckpt_output_dir=ckpt_output_dir,
	data_output_dir=processed_data_dir,                       # processed data ids/labels/masks
	max_seq_len=20000,
	max_batch_len=30000,                                      # max tokens per gpu
	num_epochs=num_epochs,
	effective_batch_size=256,                                 # target batch size per model update
	save_samples=0,                                           # save ckpt after num of samples seen (0=off)
	learning_rate=2e-5,
	warmup_steps=25,
	checkpoint_at_epoch=True,                                 # save ckpt after every epoch
	accelerate_full_state_at_epoch=False,                     # save full-state for resuming
	fsdp_options=FSDPOptions(cpu_offload_params=False),
	distributed_backend=DistributedBackend.FSDP,
	process_data=process_data,                                # can set to false if data processed before
)

# %% [markdown]
# Finally, we kick off SFT via the run_training function:

# %%
print("Start training", flush=True)

run_training(torch_args=torch_args,train_args=train_args)

print("Finished training", flush=True)

# %% [markdown]
# Upon completion, we have n (n=num_epochs) Huggingface-Format checkpoints in `experiments/training_output/hf_format`. The full run logs and metrics will also be recorded in `experiments/training_output`. Running the final training as a python script rather than in a notebook may help with progress bar writing to stdout.

# %% [markdown]
# ## Interpolation

# %%
import glob
import os

# find trained model
ckpt_dirs = glob.glob(f"{ckpt_output_dir}/hf_format/samples_*")
samples_len = len("samples_")
# print(ckpt_dirs)
max_num_samples = -1
trained_model_path = None
for ckpt_dir in ckpt_dirs:
    if not os.path.isdir(ckpt_dir):
        continue
    # print(ckpt_dir)
    num_samples_str = os.path.basename(ckpt_dir)[samples_len:]
    # print(num_samples_str)
    try:
        num_samples = int(num_samples_str)
    except ValueError:
        continue
    if max_num_samples < num_samples:
        max_num_samples = num_samples
        trained_model_path = ckpt_dir

if trained_model_path is not None:
    from interpolator import interpolate_models

    print(f"Trained model path: {trained_model_path}")

    interpolated_model_path = interpolate_models(model_path, trained_model_path)

    print(f"Interpolated model path: {interpolated_model_path}")


