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
#   --dtype auto (auto is default. specify float16 on V100)
#   --api-key dummy (if not specified, api-key is not checked)
#   --max-model-len 16384 (to avoid the following error. ValueError: The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (83552). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.)
#

# vllm serve microsoft/phi-4 --served-model-name microsoft/phi-4
#model="microsoft/phi-4"
#base_url="http://0.0.0.0:8000/v1"

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
#model="granite-3.3-8b-instruct"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-granite-3.3-8b-instruct-messages_data_teigaku-genzei-no_chat_tmpl/hf_format/samples_41966 --served-model-name granite-3.3-8b-instruct-3epochs
#model="granite-3.3-8b-instruct-3epochs"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve experiments/training_output-granite-3.3-8b-instruct-messages_data_teigaku-genzei/hf_format/samples_42052 --served-model-name granite-3.3-8b-instruct-chat-3epochs
#model="granite-3.3-8b-instruct-chat-3epochs"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-interp --served-model-name granite-3.3-8b-instruct-teigaku-genzei-interp
#model="granite-3.3-8b-instruct-teigaku-genzei-interp"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-ibm-v0 --served-model-name granite-3.3-8b-instruct-teigaku-genzei-ibm-v0
#model="granite-3.3-8b-instruct-teigaku-genzei-ibm-v0"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-ibm-v0-interp --served-model-name granite-3.3-8b-instruct-teigaku-genzei-ibm-v0-interp
#model="granite-3.3-8b-instruct-teigaku-genzei-ibm-v0-interp"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-ibm-v2 --served-model-name granite-3.3-8b-instruct-teigaku-genzei-ibm-v2
#model="granite-3.3-8b-instruct-teigaku-genzei-ibm-v2"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-ibm-v2-interp --served-model-name granite-3.3-8b-instruct-teigaku-genzei-ibm-v2-interp
#model="granite-3.3-8b-instruct-teigaku-genzei-ibm-v2-interp"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-ibm-v3 --served-model-name granite-3.3-8b-instruct-teigaku-genzei-ibm-v3
#model="granite-3.3-8b-instruct-teigaku-genzei-ibm-v3"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-ibm-v3-interp --served-model-name granite-3.3-8b-instruct-teigaku-genzei-ibm-v3-interp
#model="granite-3.3-8b-instruct-teigaku-genzei-ibm-v3-interp"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-ibm-v4-d5 --served-model-name granite-3.3-8b-instruct-teigaku-genzei-ibm-v4-d5
#model="granite-3.3-8b-instruct-teigaku-genzei-ibm-v4-d5"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-teigaku-genzei-ibm-v4-d5-interp --served-model-name granite-3.3-8b-instruct-teigaku-genzei-ibm-v4-d5-interp
#model="granite-3.3-8b-instruct-teigaku-genzei-ibm-v4-d5-interp"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-ibm-newsroom-d5 --served-model-name granite-3.3-8b-instruct-ibm-newsroom-d5
#model="granite-3.3-8b-instruct-ibm-newsroom-d5"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-ibm-newsroom-d5-interp --served-model-name granite-3.3-8b-instruct-ibm-newsroom-d5-interp
#model="granite-3.3-8b-instruct-ibm-newsroom-d5-interp"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-ibm-newsroom-d5-x100 --served-model-name granite-3.3-8b-instruct-ibm-newsroom-d5-x100
#model="granite-3.3-8b-instruct-ibm-newsroom-d5-x100"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-ibm-newsroom-d5-x100-interp --served-model-name granite-3.3-8b-instruct-ibm-newsroom-d5-x100-interp
#model="granite-3.3-8b-instruct-ibm-newsroom-d5-x100-interp"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-ibm-newsroom-en_d5 --served-model-name granite-3.3-8b-instruct-ibm-newsroom-en_d5
#model="granite-3.3-8b-instruct-ibm-newsroom-en_d5"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-ibm-newsroom-en_d5-interp --served-model-name granite-3.3-8b-instruct-ibm-newsroom-en_d5-interp
#model="granite-3.3-8b-instruct-ibm-newsroom-en_d5-interp"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-jfe-technical-report_r5 --served-model-name granite-3.3-8b-instruct-jfe-technical-report_r5
#model="granite-3.3-8b-instruct-jfe-technical-report_r5"
#base_url="http://0.0.0.0:8000/v1"

# vllm serve checkpoints/granite-3.3-8b-instruct-jfe-technical-report_r5-interp --served-model-name granite-3.3-8b-instruct-jfe-technical-report_r5-interp
model="granite-3.3-8b-instruct-jfe-technical-report_r5-interp"
base_url="http://0.0.0.0:8000/v1"


#prompt="Hello!"
#prompt="令和６年分所得税の定額減税の対象者は誰ですか？"
#prompt="令和６年分所得税の定額減税に関する情報の基礎となる法律や通達はいつのものですか？"
#prompt="合計所得金額が1,805万円を超える人は定額減税の対象となりますか？"
#{"messages":[{"content":"合計所得金額が1,805万円を超える人は定額減税の対象となりますか？","role":"user"},{"content":"いいえ、合計所得金額が1,805万円を超える人は定額減税の対象とはなりません。\n\n","role":"assistant"}]}
##prompt="令和6年分の所得税における定額減税額とは何ですか？"
##{"messages":[{"content":"令和6年分の所得税における定額減税額とは何ですか？","role":"user"},{"content":"令和6年分の所得税における定額減税額は、所得税から控除できる金額で、所得者本人には3万円が控除され、同一生計配偶者や扶養親族1人につき3万円が加算されます。\n\n","role":"assistant"}]}
##prompt="令和6年分の所得税における「定額減税額」はどのように計算されますか？"
###{"messages":[{"content":"令和6年分の所得税における「定額減税額」はどのように計算されますか？","role":"user"},{"content":"令和6年分の所得税における「定額減税額」は、所得者本人に対して3万円を基本とし、同一生計配偶者または扶養親族1人につき3万円を加算して計算されます。\n\n","role":"assistant"}]}
#prompt="IBM Quantum System Twoの設置を委託した、経済産業省所管の組織の名前は？"
#prompt="IBM と理化学研究所（理研）が設置した量子コンピューターの名前は？"
#prompt="IBM Heronプロセッサーは何量子ビットを持っていますか？"
#prompt="「Japan–IBM Quantum Partnership」の目的は何ですか？"
#prompt="「Japan–IBM Quantum Partnership」はいつ設立されましたか？"
#prompt="IBM Quantum Heronプロセッサーの量子ビット数はいくつですか？"
#prompt="Watsonx Orchestrate includes pre-built domain-specific agents. Which business areas were highlighted at IBM Think 2025?"
prompt="JFEスチールが製造している鋼管の種類にはどのようなものがありますか?"
#answer="JFEスチールは、継目無鋼管、各種溶接鋼管（ステンレス油井管、大径電縫鋼管マイティーシーム®、UOE鋼管、高機能高加工性電縫鋼管）などを製造しています。"

messages = [{"role": "user", "content": prompt}]

client = OpenAI(
    api_key="dummy",
    base_url=base_url,
    default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
)
#completion = client.completions.create(
#    model=model,
#    prompt=prompt,
#    max_tokens=1000,
#)
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=1000,
)
print(completion.to_json())
