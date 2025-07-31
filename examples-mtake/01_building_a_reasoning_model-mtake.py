# %% [markdown]
# # Building a model with fine-tuning and interpolation

# %% [markdown]
# ## Environment variables

# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['NCCL_DEBUG'] = "INFO"

# %% [markdown]
# ## Configure model

# %%
# model_path = "microsoft/Phi-4-mini-instruct"
# model_path = "checkpoints/granite-3.1-8b-lab-v1"
# model_path = "checkpoints/granite-3.1-8b-lab-v2_rev-2"
model_path = "ibm-granite/granite-3.3-8b-instruct"

model_name = os.path.basename(model_path)

# %% [markdown]
# ## Configure data
# 
# Configure `data_name` such that the message data file is `message_data_${data_name}.jsonl`.

# %%
# data_name = "nemotron"
# data_name = "teigaku-genzei"  # 14187 samples
# data_name = "teigaku-genzei-ibm_generic_tmpl"  # 14187 samples
# data_name = "teigaku-genzei-ibm-v0"
# data_name = "teigaku-genzei-ibm-v2"
# data_name = "teigaku-genzei-ibm-v3"
# data_name = "teigaku-genzei-ibm-v4-d5"
# data_name = "teigaku-genzei-ibm-v5_d5"
# data_name = "data_teigaku-genzei-ibm-v6_d5"
# data_name = "ibm-newsroom-d5"
# data_name = "ibm-newsroom-d5-x100"
# data_name = "ibm-newsroom-en_d5"  # 699 samples
data_name = "jfe-technical-report_r5"

_data_name = f"_{data_name}" if data_name is not None and len(data_name) > 0 else ""

messages_data_path = f"messages_data{_data_name}.jsonl"

force_process_data = False

# %% [markdown]
# ## Configure data preparation

# %%
num_proc = 8

force_prep_data = False

# %% [markdown]
# ## Configure fine-tuning

# %%
if data_name == "ibm-newsroom-en_d5":
    # 699 samples
    num_epochs = 100
    save_samples = 10000
    keep_last_checkpoint_only = True
else:
    # original
    num_epochs = 3
    save_samples = 0
    keep_last_checkpoint_only = False

# %% [markdown]
# ## Configure interpolation

# %%
trained_model_weight = 0.5

# %% [markdown]
# ## Data preparation

# %%
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
    assistant = sample["output"].replace("<|eot_id|>", "")
    message_list = [
        {"role": "system", "content": f"detailed thinking {sample['reasoning']}"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return {"messages": message_list}

if prep_data:
    generic_samples_datasets = []
    for split in dataset.keys():
        print(f"Processing {split} samples", flush=True)
        new_split = dataset[split].filter(
            lambda sample: sample["used_in_training"] == "yes", num_proc=num_proc
        )
        print(f"Adding {len(new_split)} samples", flush=True)
        new_samples = new_split.map(
            generalize_sample, remove_columns=list(new_split[0].keys()), num_proc=num_proc
        )
        generic_samples_datasets.append(new_samples)
        print("Samples added\n", flush=True)

# %% [markdown]
# Once we’ve got all of our reduced, generalized samples, we can re-combine them into a single dataset and save as a jsonl:

# %%
if prep_data:
    print("Writing generic messages-format data", flush=True)
    generic_samples = concatenate_datasets(generic_samples_datasets)
    print(generic_samples, flush=True)
    generic_samples.to_json(messages_data_path, lines=True, orient="records", num_proc=num_proc)
    print("Write complete!", flush=True)

# %% [markdown]
# This leaves us with 1.7 million samples of math, science, code, chat, and safety. This includes examples with and without detailed reasoning. With this file, we are ready to start SFT.

# %% [markdown]
# ## Fine-tuning

# %%
import torch

assert torch.cuda.is_available()
nproc_per_node = torch.cuda.device_count()
print(f"nproc_per_node: {nproc_per_node}", flush=True)

nnodes = 1
print(f"nnodes: {nnodes}", flush=True)

# %%
chat_tmpl_dir = "../src/instructlab/training/chat_templates"
if "granite" in model_name:
    chat_tmpl_path = f"{chat_tmpl_dir}/ibm_generic_tmpl.py"
else:
    chat_tmpl_path = None

ckpt_output_dir = f"experiments/training_output-{model_name}{_data_name}"
processed_data_dir = f"data/processed-data-{model_name}{_data_name}"

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
from instructlab.training.config import (
    TorchrunArgs,
    TrainingArgs,
    DistributedBackend,
    FSDPOptions,
)
from instructlab.training.main_ds import run_training

# %% [markdown]
# We then define our distributed settings via TorchrunArgs. In our case, we trained on a single node with 8 H100 GPUs:

# %%
torch_args = TorchrunArgs(
    nproc_per_node=nproc_per_node,
    nnodes=nnodes,
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
    data_output_dir=processed_data_dir,  # processed data ids/labels/masks
    max_seq_len=20000,
    max_batch_len=30000,  # max tokens per gpu
    num_epochs=num_epochs,
    effective_batch_size=256,  # target batch size per model update
    learning_rate=2e-5,
    warmup_steps=25,
    save_samples=save_samples,  # save ckpt after num of samples seen (0=off)
    checkpoint_at_epoch=True,  # save ckpt after every epoch
    accelerate_full_state_at_epoch=False,  # save full-state for resuming
    process_data=process_data,  # can set to false if data processed before
    keep_last_checkpoint_only=keep_last_checkpoint_only,
    distributed_backend=DistributedBackend.FSDP,
    fsdp_options=FSDPOptions(cpu_offload_params=False),
)

# %% [markdown]
# Finally, we kick off SFT via the run_training function:

# %%
print("Start training", flush=True)

run_training(torch_args=torch_args,train_args=train_args)

print("Finished training", flush=True)

# %% [markdown]
# Upon completion, we have `{num_epochs}` Huggingface-Format checkpoints in `{ckpt_output_dir}/hf_format`. The full run logs and metrics will also be recorded in `{ckpt_output_dir}`. Running the final training as a python script rather than in a notebook may help with progress bar writing to stdout.

# %% [markdown]
# ## Interpolation
# 
# When the training is completed successfully, we will interpolate the last checkpoint with the original model to recover the capability that may have been lost during the training process. `{output_model_path}` will be `{trained_model_path}-interp` by default.
# 
# We can also interpolate models manually as follows.
# ```sh
# python interpolator.py --model_path {model_path} --trained_model_path {trained_model_path} --trained_model_weight {trained_model_weight}
# ```

# %%
import glob

def find_last_checkpoint(ckpt_output_dir: str) -> str | None:
    last_checkpoint_path = None

    # For keep_last_checkpoint_only is True
    # See https://github.com/instructlab/training/blob/4eb4173f2508dc1fd8db7e30b59609f0ceeb25ac/src/instructlab/training/config.py#L229
    ckpt_dirs = glob.glob(f"{ckpt_output_dir}/hf_format/last_epoch")
    for ckpt_dir in ckpt_dirs:
        last_checkpoint_path = ckpt_dir

    # For keep_last_checkpoint_only is False
    if last_checkpoint_path is None:
        ckpt_dirs = glob.glob(f"{ckpt_output_dir}/hf_format/samples_*")
        samples_len = len("samples_")
        max_num_samples = -1
        for ckpt_dir in ckpt_dirs:
            if not os.path.isdir(ckpt_dir):
                continue
            num_samples_str = os.path.basename(ckpt_dir)[samples_len:]
            try:
                num_samples = int(num_samples_str)
            except ValueError:
                continue
            if max_num_samples < num_samples:
                max_num_samples = num_samples
                last_checkpoint_path = ckpt_dir

    return last_checkpoint_path

# %%
trained_model_path = find_last_checkpoint(ckpt_output_dir)

if trained_model_path is not None:
    from interpolator import interpolate_models

    print(f"Trained model path: {trained_model_path}")

    output_model_path = interpolate_models(model_path, trained_model_path, trained_model_weight=trained_model_weight)

    print(f"Output model path: {output_model_path}")


