#!/u/gakuto/anaconda3/bin/python
# -*- coding: utf-8 -*-

"""
Time-stamp: <2024-11-25 21:39:26 gakuto>
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, glob, gzip, argparse
from collections import OrderedDict
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_arguments():
    parser = argparse.ArgumentParser(description='XXX')

    parser.add_argument('--ckpts_in', type=str, default='',
                    help='a:0.5,b:0.5')

    parser.add_argument('--ckpt_out', type=str, default='',
                    help='out')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    new_state_dict = OrderedDict()

    l_ckpt_weight = args.ckpts_in.split(',')
    
    for index, ckpt_weight in enumerate(l_ckpt_weight):
        ckpt, weight = ckpt_weight.split(':')
        weight = float(weight)

        print(ckpt)
        print(weight)

        if index == 0:
            model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            interp_model_state_dict = model.state_dict()

            for key in interp_model_state_dict.keys():
                interp_model_state_dict[key] = interp_model_state_dict[key] * weight
        else:
            model_tmp = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
            interp_model_state_dict_tmp = model_tmp.state_dict()
            for key in interp_model_state_dict.keys():
                interp_model_state_dict[key] += interp_model_state_dict_tmp[key] * weight

    # save
    model.save_pretrained(args.ckpt_out, state_dict=interp_model_state_dict)
    tokenizer.save_pretrained(args.ckpt_out)

    #training_config = json.load(open(os.path.join(l_ckpt_weight[0].split(':')[0], "training_config.json"), "r"))
    #json.dump(training_config, open(os.path.join(args.ckpt_out, "training_config.json"), "w"), indent=4)
    
            
if __name__ == "__main__":
    main()
