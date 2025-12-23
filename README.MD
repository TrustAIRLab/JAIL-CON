# $\texttt{JAIL-CON}$


This is the anonymous repository of the submission "Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency."

*Be careful! This repository may contain potentially unsafe information. User discretion is advised.*

## How to use this repository?

Too long to read, would like to see the attack results directly? Please see [Step D](#d-show-evaluation-results).

### A. Install and set the ENV
1. Clone this repository.
2. Prepare the python ENV.
```
conda create -n jailcon python=3.12 -y
conda activate jailcon
cd PATH_TO_THE_REPOSITORY
bash prepare.sh
```

### B. Jailbreak LLMs
```
python Parallel_QA.py \
--llm_model $model \
--jailbreak_method $jailbreak_method \
--separator $separator \
--openai_key $openai_key \
--huggingface_key $huggingface_key \
--deepseek_key $deepseek_key \
--max_queries $max_queries
```
```$model``` is the target LLM, which could be assigned as LLaMA2-13B, Vicuna-13B, Mistral-7B, LLaMA3-8B, GPT-4o, or DeepSeek-V3.

```$jailbreak_method``` is the used variant in $\texttt{JAIL-CON}$, set Parallel_Auto1 for CIT (concurrency with idle task) and Parallel_Auto2 for CVT (concurrency with valid task).

```$separator``` is the key for the selected separator among {"A": "{}", "B": "<>", "C": "[]", "D": "$$", "E": "##", "F": "😊😊"}.

```$xx_key``` is the api key or token for the corresponding platform.

```$max_queries``` is the max number of attack iterations.

For instance, to launch an attack against GPT-4o with default settings, you should run two scripts below.
For CIT, you run
```
python Parallel_QA.py \
--llm_model GPT-4o \
--jailbreak_method Parallel_Auto1 \
--separator A \
--openai_key $openai_key \
--max_queries 50
```
For CVT, you run
```
python Parallel_QA.py \
--llm_model GPT-4o \
--jailbreak_method Parallel_Auto2 \
--separator A \
--openai_key $openai_key \
--max_queries 50
```

### C. Evaluate Attack
```
python Eval.py \
--llm_model $model \
--jailbreak_method $jailbreak_method \
--separator $separator \
--openai_key $openai_key \
--eval_model $eval_model
```
```$eval_model``` is the selected model for evluation, where 'GPT-4o' is used for judging whether an answer is successful and 'Moderation' is used for filtering.

Then there will be a json file with the file name starting with 'Safety_GPT-4o_xx'. The 'Original_Safety' in the file determines whether an answer is successful, and the 'Moderation_Flag' determines whether the answer will be filtered by the guardrail.

### D. Show Evaluation Results
```
python Eval_Results.py \
--separator $separator \
```
For demonstration, we have provided the main results (for ```$separator=A```) in directory ```/Dataset/```.

Then, the results will be given as
```
LLM: GPT-4o, ASR-O: 0.95, Filter_Rate: 0.2, ASR-E: 0.76
LLM: DeepSeek-V3, ASR-O: 0.95, Filter_Rate: 0.3684210526315789, ASR-E: 0.6
LLM: LLaMA2-13B, ASR-O: 0.86, Filter_Rate: 0.27906976744186046, ASR-E: 0.62
LLM: LLaMA3-8B, ASR-O: 1.0, Filter_Rate: 0.44, ASR-E: 0.56
LLM: Mistral-7B, ASR-O: 0.96, Filter_Rate: 0.3541666666666667, ASR-E: 0.6199999999999999
LLM: Vicuna-13B, ASR-O: 0.97, Filter_Rate: 0.30927835051546393, ASR-E: 0.6699999999999999
```
