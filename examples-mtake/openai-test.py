from openai import OpenAI
import os

#model="ibm-granite/granite-3.0-8b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct/v1"

#model="ibm-granite/granite-3.1-8b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1"

#model="ibm-granite/granite-3.3-8b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-instruct/v1"

#model="meta-llama/llama-3-1-70b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-70b-instruct/v1"

#model="meta-llama/llama-3-3-70b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1"

#model="microsoft/phi-4"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4/v1"

#
# pip install vllm 'flashinfer-python<0.2.3'
#
# vllm serve <model_name>
# (Equivalent to: python -m vllm.entrypoints.openai.api_server --model <model_name>)
# Options:
#   --dtype auto (auto is default)
#   --api-key dummy (if not specified, api-key is not checked)
#

# vllm serve microsoft/Phi-4-mini-instruct --served-model-name Phi-4-mini-instruct
#model="Phi-4-mini-instruct"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-Phi-4-mini-instruct-messages_data_teigaku-genzei/hf_format/samples_14021 --served-model-name Phi-4-mini-instruct-1epoch
#model="Phi-4-mini-instruct-1epoch"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve /u/mtake/.cache/instructlab/models/granite-3.1-8b-lab-v1 --served-model-name granite-3.1-8b-lab-v1
#model="granite-3.1-8b-lab-v1"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-granite-3.1-8b-lab-v1-messages_data_teigaku-genzei-no_chat_tmpl/hf_format/samples_42061 --served-model-name granite-3.1-8b-lab-v1-3epochs
#model="granite-3.1-8b-lab-v1-3epochs"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-granite-3.1-8b-lab-v1-messages_data_teigaku-genzei/hf_format/samples_42052 --served-model-name granite-3.1-8b-lab-v1-chat-3epochs
#model="granite-3.1-8b-lab-v1-chat-3epochs"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve /u/mtake/.cache/instructlab/models/granite-3.1-8b-lab-v2_rev-2 --served-model-name granite-3.1-8b-lab-v2_rev-2
#model="granite-3.1-8b-lab-v2_rev-2"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-granite-3.1-8b-lab-v2_rev-2-messages_data_teigaku-genzei-no_chat_tmpl/hf_format/samples_41913 --served-model-name granite-3.1-8b-lab-v2_rev-2-3epochs
#model="granite-3.1-8b-lab-v2_rev-2-3epochs"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-granite-3.1-8b-lab-v2_rev-2-messages_data_teigaku-genzei/hf_format/samples_42052 --served-model-name granite-3.1-8b-lab-v2_rev-2-chat-3epochs
#model="granite-3.1-8b-lab-v2_rev-2-chat-3epochs"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve ibm-granite/granite-3.3-8b-instruct --served-model-name granite-3.3-8b-instruct
model="granite-3.3-8b-instruct"
base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-granite-3.3-8b-instruct-messages_data_teigaku-genzei-no_chat_tmpl/hf_format/samples_41966 --served-model-name granite-3.3-8b-instruct-3epochs
#model="granite-3.3-8b-instruct-3epochs"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-granite-3.3-8b-instruct-messages_data_teigaku-genzei/hf_format/samples_42052 --served-model-name granite-3.3-8b-instruct-chat-3epochs
#model="granite-3.3-8b-instruct-chat-3epochs"
#base_url="http://0.0.0.0:8000/v1"

#prompt="Hello!"
#prompt="令和６年分所得税の定額減税の対象者は誰ですか？"
prompt="令和６年分所得税の定額減税に関する情報の基礎となる法律や通達はいつのものですか？"

client = OpenAI(
    api_key="dummy",
    base_url=base_url,
    default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
)
completion = client.completions.create(
    model=model,
    prompt=prompt,
    max_tokens=1000,
)
print(completion.to_json())
