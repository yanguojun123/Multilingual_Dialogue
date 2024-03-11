from peft import PeftConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
import json
import re
import argparse
import logging
from langchain.llms import HuggingFacePipeline


from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn as nn
import torch.optim as optim

parameter_ids = {'<en> <en>':0,'<de> <de>':1,'<it> <it>':2,
                 '<en> <de>':3,'<en> <it>':4, '<de> <en>':5,
                 '<de> <it>':6,'<it> <en>':7,'<it> <de>':8}

# parameter_ids = {'<en> <en>': 0, '<es> <es>': 1, '<th> <th>': 1, '<en> <es>': 3, '<en> <th>': 3,
#                  '<es> <en>': 5, '<es> <th>': 3, '<th> <en>': 5, '<th> <es>': 5}


class TripleMolrCausalLM(nn.Module):
    def __init__(self, model_id):
        super(TripleMolrCausalLM, self).__init__()
        self.model_id = model_id

        # Load pre-trained model and config
        config = AutoConfig.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, config=config)

        # LLM + Task layer
        self.base_model = self.model.transformer
        # Freeze the parameters of the base model
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # Define the additional layer
        self.mapping = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mapping_1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mapping_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mapping_3 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mapping_4 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mapping_5 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Set requires_grad to True for the new layer
        # for param in self.new_layer.parameters():
        #     param.requires_grad = True

        # Identify the task layer
        self.lm_head = self.model.lm_head  # Assuming lm_head is the task layer in GPT-like models
        self.lm_head_1 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head_2 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Set requires_grad to True for the task layer
        # for param in self.lm_head.parameters():
        #     param.requires_grad = True

        # Modify the model to include the new layer before the task layer
        # self.model.transformer.add_module('mapping', self.mapping) # dynamtically added
        # self.base_model.transformer.h += 1  # Increment the number of layers
        config.n_layer += 1

    def forward(self, input_ids, **kwargs):
        sequence_output = self.base_model(input_ids, **kwargs)
        parameter_id = kwargs['parameter_id']
        print("parameter_id:", parameter_id)
        if parameter_id<3:              ### Monolingual without mapping
            if parameter_id == 0:       # '<en> <en>'
              lm_logits = self.lm_head(sequence_output) # [B, L, V]
            elif parameter_id == 1:     # '<de> <de>'
              lm_logits = self.lm_head_1(sequence_output) # [B, L, V]
            elif parameter_id == 2:     # '<it> <it>'
              lm_logits = self.lm_head_2(sequence_output) # [B, L, V]
        elif parameter_id < 5:          ### Pivot language (LLM) <en>
            if parameter_id == 3:       # '<en> <de>'
                lmp_logits = self.mapping_2(sequence_output) #[B, L, H] * [H, H]
            elif parameter_id == 4:     # <en> <it>'
                lmp_logits = self.mapping_1(sequence_output) #[B, L, H] * [H, H]
            lm_logits = self.lm_head_1(lmp_logits) # [B, L, V]
        elif parameter_id < 7:          ### Pivot language (LLM) <de>
            if parameter_id == 5:       # '<de> <en>'
                lmp_logits = self.mapping(sequence_output) #[B, L, H] * [H, H]
            elif parameter_id == 6:     # '<de> <it>'
                lmp_logits = self.mapping_3(sequence_output) #[B, L, H] * [H, H]
            lm_logits = self.lm_head(lmp_logits) # [B, L, V]
        else:                           ### Pivot language (LLM) <it>
            if parameter_id == 7:       # '<it> <en>'
                lmp_logits = self.mapping_4(sequence_output) #[B, L, H] * [H, H]
            elif parameter_id == 8:     # '<it> <de>'
                lmp_logits = self.mapping_5(sequence_output) #[B, L, H] * [H, H]
            lm_logits = self.lm_head_2(lmp_logits) # [B, L, V]
        return lm_logits


if __name__ == '__main__':
    # Example usage:
    model_id = "FreedomIntelligence/phoenix-inst-chat-7b"
    model = TripleMolrCausalLM(model_id)