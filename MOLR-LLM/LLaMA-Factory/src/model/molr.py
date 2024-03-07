from peft import PeftConfig
from transformers import MT5ForConditionalGeneration, AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn as nn

parameter_ids = {'<en> <en>':0,'<de> <de>':1,'<it> <it>':2,
                 '<en> <de>':3,'<en> <it>':4, '<de> <en>':5,
                 '<de> <it>':6,'<it> <en>':7,'<it> <de>':8}

# parameter_ids = {'<en> <en>': 0, '<es> <es>': 1, '<th> <th>': 1, '<en> <es>': 3, '<en> <th>': 3,
#                  '<es> <en>': 5, '<es> <th>': 3, '<th> <en>': 5, '<th> <es>': 5}


class TripleMolrCausalLM(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)

        # Define the additional layerDefine the additional layer
        self.mapping = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)    # <de> <en> mapping
        self.mapping_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)  # <it> <de> mapping
        # self.mapping_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)  # <en> <de> mapping ==> reuse self.mapping
        self.mapping_3 = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)  # <it> <en> mapping
        # self.mapping_4 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)  # <en> <it> mapping ==> reuse self.mapping_3
        # self.mapping_5 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)  # <de> <it> mapping ==> reuse self.mapping_1


    def forward(self, input_ids, **kwargs):
        route_id = input_ids[0][1] # Set the second token (excluding special token <s>) of the first instruction as the id for language routing
        route_token = self.tokenizer.decode(route_id) # Convert to original token
        input_ids = [seq[1:] for seq in input_ids]

        outputs = self.transformer(input_ids, **kwargs)
        last_hidden_state = outputs.last_hidden_state

        # print("route_token:", route_token)
        ### Monolingual without mapping
        if route_token in ['<r0>', '<r1>', '<r2>']:       # '<en> <en>'
            lm_logits = self.lm_head(outputs) # [B, L, V]
        ### Multilingual with mapping
        if route_token in ['<r3>', '<r5>']:       # '<r3> <en> <de>', '<r5> <de> <en>'
            custom_output = self.mapping(last_hidden_state) #[B, L, H] * [H, H]
            lm_logits = self.lm_head(custom_output)  # [B, L, V]
        elif route_token in ['<r4>', '<r8>']:     # '<r4> <it> <de>', '<r8> <de> <it>'
            custom_output = self.mapping_1(last_hidden_state) #[B, L, H] * [H, H]
            lm_logits = self.lm_head(custom_output) # [B, L, V]
        elif route_token in ['<r6>', '<r7>']:     # '<r6> <it> <en>', '<r7> <en> <it>'
            custom_output = self.mapping_3(last_hidden_state) #[B, L, H] * [H, H]
            lm_logits = self.lm_head(custom_output) # [B, L, V]
        else:
            pass

        return lm_logits

    def generate(self, prompt, tokenizer, max_length=50, num_return_sequences=1):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # Use the generate method from AutoModelForCausalLM
        output = super().generate(input_ids=input_ids,
                                  max_length=max_length,
                                  num_return_sequences=num_return_sequences,
                                  pad_token_id=tokenizer.eos_token_id)
        return output


class TripleMolrConditionalCausalLM(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)

        # Define the additional layerDefine the additional layer
        self.mapping = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)    # <de> <en> mapping
        self.mapping_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)  # <it> <de> mapping
        # self.mapping_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)  # <en> <de> mapping ==> reuse self.mapping
        self.mapping_3 = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)  # <it> <en> mapping
        # self.mapping_4 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)  # <en> <it> mapping ==> reuse self.mapping_3
        # self.mapping_5 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)  # <de> <it> mapping ==> reuse self.mapping_1


    def forward(self, input_ids, **kwargs):
        route_id = input_ids[0][1] # Set the second token (excluding special token <s>) of the first instruction as the id for language routing
        route_token = self.tokenizer.decode(route_id) # Convert to original token
        input_ids = [seq[1:] for seq in input_ids]

        outputs = self.model(input_ids, **kwargs)
        last_hidden_state = outputs.last_hidden_state

        # print("route_token:", route_token)
        ### Monolingual without mapping
        if route_token in ['<r0>', '<r1>', '<r2>']:       # '<en> <en>'
            lm_logits = self.lm_head(outputs) # [B, L, V]
        ### Multilingual with mapping
        if route_token in ['<r3>', '<r5>']:       # '<r3> <en> <de>', '<r5> <de> <en>'
            custom_output = self.mapping(last_hidden_state) #[B, L, H] * [H, H]
            lm_logits = self.lm_head(custom_output)  # [B, L, V]
        elif route_token in ['<r4>', '<r8>']:     # '<r4> <it> <de>', '<r8> <de> <it>'
            custom_output = self.mapping_1(last_hidden_state) #[B, L, H] * [H, H]
            lm_logits = self.lm_head(custom_output) # [B, L, V]
        elif route_token in ['<r6>', '<r7>']:     # '<r6> <it> <en>', '<r7> <en> <it>'
            custom_output = self.mapping_3(last_hidden_state) #[B, L, H] * [H, H]
            lm_logits = self.lm_head(custom_output) # [B, L, V]
        else:
            pass

        return lm_logits

    def generate(self, prompt, tokenizer, max_length=50, num_return_sequences=1):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # Use the generate method from AutoModelForCausalLM
        output = super().generate(input_ids=input_ids,
                                  max_length=max_length,
                                  num_return_sequences=num_return_sequences,
                                  pad_token_id=tokenizer.eos_token_id)
        return output