from openai import OpenAI
import os

#model="ibm-granite/granite-3.0-8b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct/v1"

#model="ibm-granite/granite-3.1-8b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1"

#model="meta-llama/llama-3-1-70b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-70b-instruct/v1"

#model="meta-llama/llama-3-3-70b-instruct"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1"

#model="microsoft/phi-4"
#base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4/v1"

#
# NOTE: assumption
# pip install vllm 'flashinfer-python<0.2.3'
# python -m vllm.entrypoints.openai.api_server --model experiments/training_output-Phi-4-mini-instruct-generic_data_teigaku-genzei/hf_format/samples_14021 --dtype auto # --api-key dummy
#
#model="experiments/training_output-Phi-4-mini-instruct-generic_data_teigaku-genzei/hf_format/samples_14021"
#base_url="http://0.0.0.0:8000/v1"


#
# NOTE: assumption
# pip install vllm 'flashinfer-python<0.2.3'
# python -m vllm.entrypoints.openai.api_server --model /u/mtake/.cache/instructlab/models/granite-3.1-8b-lab-v1 --dtype auto # --api-key dummy
#
#model="/u/mtake/.cache/instructlab/models/granite-3.1-8b-lab-v1"
#base_url="http://0.0.0.0:8000/v1"


# (training-py311) [mtake@cccxc596 examples-mtake]$ ls experiments/training_output-granite-3.1-8b-lab-v1-generic_data_teigaku-genzei/hf_format | cat
# samples_14027
# samples_28049
# samples_42061
#
# NOTE: assumption
# pip install vllm 'flashinfer-python<0.2.3'
# python -m vllm.entrypoints.openai.api_server --model experiments/training_output-granite-3.1-8b-lab-v1-generic_data_teigaku-genzei/hf_format/samples_14027 --dtype auto # --api-key dummy
# python -m vllm.entrypoints.openai.api_server --model experiments/training_output-granite-3.1-8b-lab-v1-generic_data_teigaku-genzei/hf_format/samples_42061 --dtype auto # --api-key dummy
#
#model="experiments/training_output-granite-3.1-8b-lab-v1-generic_data_teigaku-genzei/hf_format/samples_14027"
model="experiments/training_output-granite-3.1-8b-lab-v1-generic_data_teigaku-genzei/hf_format/samples_42061"
base_url="http://0.0.0.0:8000/v1"

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
