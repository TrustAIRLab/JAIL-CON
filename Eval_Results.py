import json
import numpy as np
import pandas as pd
import argparse
import operator


def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference Script")
    
    parser.add_argument('--separator', type=str, default='A', help='A: { }, B: < >, C: [ ], D: $ $, E: # #, F: 😊 😊')

    args = parser.parse_args()
    return args


def find_items_by_multiple_key_values(json_data, conditions):
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    if not isinstance(json_data, list):
        raise ValueError("JSON should be a list")

    op_map = {
        '==': operator.eq,
        '!=': operator.ne,
        '<=': operator.le,
        '>=': operator.ge,
        '<': operator.lt,
        '>': operator.gt
    }

    def parse_condition(key, value):
        for op_str in ['<=', '>=', '!=', '<', '>', '==']:
            if key.endswith(f' {op_str}'):
                real_key = key[:-len(f' {op_str}')]
                return lambda item: op_map[op_str](item.get(real_key), value)
        return lambda item: item.get(key) == value

    condition_funcs = [parse_condition(k, v) for k, v in conditions.items()]

    matching_indices = [
        index for index, item in enumerate(json_data)
        if isinstance(item, dict) and all(cond(item) for cond in condition_funcs)
    ]

    return len(matching_indices), matching_indices


args = parse_args()
separator_idx = args.separator
llm_models = ['GPT-4o', 'DeepSeek-V3', 'LLaMA2-13B', "LLaMA3-8B", 'Mistral-7B', 'Vicuna-13B']
judge_model = 'StrongReject'
jailbreak_method = 'Parallel_Auto'
dataset_path = 'Dataset'

for llm_model in llm_models:
    if jailbreak_method == 'Parallel_Auto':
            ASRs = []
            delta_ASRs = []
            original_unsafe_indexes = []
            original_unsafe_unfiltered_indexes = []
            for idx_basic in range(1, 3):
                file_path = f"{dataset_path}/Safety_GPT-4o_JBB-Behaviors_{llm_model}_{jailbreak_method}{str(idx_basic)}_{judge_model}_{separator_idx}.json"
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    conditions = {'Original_Safety': 'unsafe'}
                    original_unsafe, indexes = find_items_by_multiple_key_values(data, conditions)
                    original_unsafe_indexes.append(indexes)
                    conditions = {'Original_Safety': 'unsafe', 'Moderation_Flag': False}
                    original_unsafe_unfiltered, indexes = find_items_by_multiple_key_values(data, conditions)
                    original_unsafe_unfiltered_indexes.append(indexes)

            union_original_unsafe_indexes = list(set().union(*original_unsafe_indexes))
            union_original_unsafe_unfiltered_indexes = list(set().union(*original_unsafe_unfiltered_indexes))
            union_ASR = len(union_original_unsafe_indexes) / 100
            # print(len(union_original_unsafe_unfiltered_indexes))
            union_filtered_rate = (len(union_original_unsafe_indexes) - len(union_original_unsafe_unfiltered_indexes)) / len(union_original_unsafe_indexes)

            print(f"LLM: {llm_model}, ASR-O: {union_ASR}, Filter_Rate: {union_filtered_rate}, ASR-E: {union_ASR * (1 - union_filtered_rate)}")
