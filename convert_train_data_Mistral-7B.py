from tqdm import tqdm
import sys
import json

import random
import argparse
import os
import torch
import re

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

def parse_args():
    parser = argparse.ArgumentParser(description='Process data with a specified data directory')
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Root directory for data files')
    return parser.parse_args()

def cls_evaluate(tok, model, txt_batch):
    results = []
    
    for text in txt_batch:
        print(f'Input text: {text}')
        input_enc = tok(
            text,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
            truncation=True,
            return_attention_mask=True
        )
        
        input_ids = input_enc.input_ids.cuda()
        attn_mask = input_enc.attention_mask.cuda()
        
        with torch.no_grad():
            # Generate text with the causal LM with increased max_new_tokens
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=512,  # Increased to avoid truncation
                do_sample=False  # Deterministic generation
            )
        
        # Decode the full output
        full_generated_text = tok.decode(output[0], skip_special_tokens=True)
        
        # Decode just the input part to find where to split
        input_text = tok.decode(input_ids[0], skip_special_tokens=True)
        
        # Extract only the newly generated part (the model's response)
        model_response = full_generated_text[len(input_text):].strip()
        
        print(f'Generated text: {model_response}')
        
        # New logic: Find all occurrences of informative/distracting keywords and use the LAST one
        model_response_lower = model_response.lower()
        
        # Create lists to store all occurrences of informative/distracting keywords
        informative_indices = []
        distracting_indices = []
        
        # Find all instances of "informative" not preceded by "not" or "isn't"
        for match in re.finditer(r'(?<!not\s)(?<!isn[\'"]t\s)informative', model_response_lower):
            informative_indices.append(match.start())
        
        # Find all instances of keywords that indicate the result is distracting
        distracting_keywords = ["distracting", "irrelevant", "not relevant", "not helpful", "not informative", "isn't informative"]
        for keyword in distracting_keywords:
            for match in re.finditer(keyword, model_response_lower):
                distracting_indices.append(match.start())
        
        # Determine final classification based on the last occurrence
        if informative_indices and distracting_indices:
            # Both types of keywords found, use the one that appears last
            last_informative = max(informative_indices) if informative_indices else -1
            last_distracting = max(distracting_indices) if distracting_indices else -1
            
            is_informative = last_informative > last_distracting
        elif informative_indices:
            # Only informative keywords found
            is_informative = True
        elif distracting_indices:
            # Only distracting keywords found
            is_informative = False
        else:
            # No clear keywords found, use fallbacks
            if re.search(r'\byes\b', model_response_lower) or "answer: yes" in model_response_lower:
                is_informative = True
            elif re.search(r'\bno\b', model_response_lower) or "answer: no" in model_response_lower:
                is_informative = False
            else:
                is_informative = "true" in model_response_lower
        
        print(f'Is informative: {is_informative}')
        print('-' * 50) 

        # Create a pseudo-logit structure
        if is_informative:
            logits = torch.tensor([[1.0, 0.0, 0.0]])
        else:
            logits = torch.tensor([[0.0, 0.0, 1.0]])
            
        results.append(logits)
    
    return torch.cat(results, dim=0)


def format_case(case):
    title = case['title']
    body = case['body']
    return f'{title}\n{body}'


def format_search_res(search_res_list, bm25_list, rand_num, tok, model, gpt_str, n = 5):
    
    if search_res_list is not None:
        search_res_list = search_res_list[:n]
        search_res_list.reverse()
        search_res_list = [
            f'{format_case(x)}' for i, x in enumerate(search_res_list)
        ]
    else:
        if len(bm25_list) == 0 or bm25_list is None:
            return None, None
        search_res_list = []

    search_res_list += bm25_list[:1]

    if rand_num < 0.2:
        return None, None
    
    elif rand_num < 0.3:
        num_res = 1
    
    elif rand_num < 0.6:
        num_res = 2
    
    else:
        num_res = 3
    
    num_res = min(num_res, len(search_res_list))
    search_res_list = random.sample(search_res_list, num_res)
    # ent_list = [
    #     f'{x} is entailed by {gpt_str}' for x in search_res_list
    # ]
    ent_list = [
        f'Search result: "{x}"\nQuestion response: "{gpt_str}"\nIs this search result informative and relevant to the question response? Answer with only "informative" if informative or "distracting" if distracting or irrelevant.' 
        for x in search_res_list
    ]

    logits = cls_evaluate(tok, model, ent_list)
    ent_pred = (logits[:, 0] > logits[:, 2]).float().tolist()

    if sum(ent_pred) == 0:
        if len(search_res_list) == 1:
            verify_str = 'The search result is distracting. I should ignore the search result and only utilize my own knowledge.'
        else:
            verify_str = 'All search results are distracting, I should ignore all search results and only utilize my own knowledge.'
    
    elif sum(ent_pred) == len(search_res_list):
        if len(search_res_list) == 1:
            verify_str = 'The search result is information, and I can utilize the search result and my knowledge.'
        else:
            verify_str = 'All search results are informative, and I can utilize all search results and my knowledge.'
    
    else:
        item_list = []
        label_list = []
        for i, ent in enumerate(ent_pred):
            if ent == 1:
                item_list.append(f'search result ({i + 1}) is informative')
                label_list.append(f'({i + 1})')
            else:
                item_list.append(f'search result ({i + 1}) is distracting')
        
        item_list[-1] = f'and {item_list[-1]}'
        itemized_ent = ', '.join(item_list).capitalize()
        label_str = ', '.join(label_list)

        if len(label_list) == 1:
            res_term = 'result'
        else:
            res_term = 'results'

        verify_str = f'{itemized_ent}. I will utilize the informative search {res_term} {label_str} and my knowledge.'

    search_res_str = '\n\n'.join([
        f'({i + 1}) {x}' for i, x in enumerate(search_res_list)
    ])

    return search_res_str, verify_str


def format_search_res_rand(search_res_all, n = 3):
    search_res_list = random.sample(search_res_all, n)

    search_res_str = '\n\n'.join(search_res_list)
    return search_res_str


def convert(data_idx, data_case, rand_num, tok, model, search_res_all=None):
    gpt_str = data_case['output']

    if search_res_all is None:
        search_res_str, verify_str = format_search_res(
            data_case['search_res'], data_case['bm25_res'], rand_num, tok, model, gpt_str, n = 6
        )
    else:
        search_res_str = format_search_res_rand(search_res_all)
    
    if data_case['input'] is None:
        data_case['input'] = ''
    data_case['input'] = data_case['input'].strip()

    if len(data_case['input']) == 0:
        if search_res_str is not None:
            human_str = f'\n{search_res_str}\n\n### Instruction: {data_case["instruction"]}'
        else:
            human_str = f'\nNone\n\n### Instruction: {data_case["instruction"]}'

    else:
        if search_res_str is not None:
            human_str = f'\n{search_res_str}\n\n### Instruction: {data_case["instruction"]}\n### Input: {data_case["input"]}'
        else:
            human_str = f'\nNone\n\n### Instruction: {data_case["instruction"]}\n### Input: {data_case["input"]}'
    
    if verify_str is not None:
        gpt_str = f'{verify_str} {gpt_str}'
    
    conversation = {
        'id': f'conv_{data_idx}',
        'conversations': [
            {
                'from': 'human',
                'value': human_str,
            },
            {
                'from': 'gpt',
                'value': gpt_str
            }
        ]
    }
    return conversation


def merge_data(data_dir):
    gpt4_data = json.load(open(os.path.join(data_dir, 'alpaca_gpt4_data.json')))
    search_data = json.load(open(os.path.join(data_dir, 'search_res_only.json')))

    for i, search_res in enumerate(search_data):
        gpt4_data[i]['search_res'] = search_res[0]
        gpt4_data[i]['bm25_res'] = search_res[1]
    
    return gpt4_data


if __name__ == '__main__':
    args = parse_args()
    num_split = 16
    dataset = []
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16).cuda()

    model.eval()

    dataset = merge_data(args.data_dir)
    random.seed(42)
    dataset = random.sample(dataset, 14000)

    rand_tensor = torch.rand(len(dataset))
    rand_table = rand_tensor.tolist()

    formatted_dataset = []
    for i, x in tqdm(enumerate(dataset), total=len(dataset), desc="Generating SAIL_train"):
        formatted = convert(i, x, rand_table[i], tokenizer, model)
        formatted_dataset.append(formatted)

    json.dump(formatted_dataset, open(os.path.join(args.data_dir, 'SAIL_train.json'), 'w'))