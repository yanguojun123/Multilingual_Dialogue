import csv
import json
import re
import pickle
#from transformers import MT5ForConditionalGeneration
#from transformers import MT5Tokenizer
import torch
import random
#from nltk.stem import WordNetLemmatizer
#from transformers import MT5Tokenizer
#from pattern.en import lemma
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import f1_score
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import os
import simplemma
# from googletrans import Translator
# translator = Translator(service_urls=['translate.google.com'])
langdata = simplemma.load_data('en', 'de', 'it', 'es')
#from utils import translate
def get_dict(src_lang, tar_lang):
    if os.path.exists('data/dst_vocab/'+src_lang+'-'+tar_lang+'.txt'):
        tokens = open('data/dst_vocab/'+src_lang+'-'+tar_lang+'.txt', 'r', encoding='utf-8').read().split('\n')[:-1]
        dicts = {}
        for x in tokens:
            key = x.split()[0]
            value = x.split()[1]
            if key in dicts.keys():
                dicts[key].append(value)
            else:
                dicts[key] = [value]

        return dicts
    else: # if none of language is English
        tokens_en = open('data/dst_vocab/' + src_lang + '-en.txt', 'r', encoding='utf-8').read().split('\n')[
                 :-1]
        tokens_target = open('data/dst_vocab/' + 'en-' + tar_lang + '.txt', 'r', encoding='utf-8').read().split('\n')[
                    :-1]
        en_dicts = {}
        src_dicts = {}
        for x in tokens_target:
            key = x.split()[0]
            value = x.split()[1]
            if key in en_dicts.keys():
                en_dicts[key].append(value)
            else:
                en_dicts[key] = [value]
        for x in tokens_en:
            key = x.split()[0]
            value = x.split()[1]
            if value in en_dicts.keys():
                if key in src_dicts.keys():
                    src_dicts[key] += en_dicts[value]
                else:
                    src_dicts[key] = en_dicts[value]
        return src_dicts

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def preproces_more_tokens(utterance,dicts):
    tokens = word_tokenize(utterance)  # 分词
    tagged_sent = pos_tag(tokens)  # 获取单词词性

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        # print(tag)
        if tag[1] == 'CD' and tag[0] not in ['one', 'two', 'three'] and len(tag[0]) >= 5:
            lemmas_sent.append('<|number|>')
            print(tag[0])
        else:
            pro_token = simplemma.lemmatize(tag[0], langdata)
            pro_token_new = pos_tag([pro_token])


            if pro_token_new[0][1] in ['JJ', 'NN', 'NNS', 'VB']:
                if pro_token_new[0][0] in dicts.keys():
                    processed_token = dicts[pro_token_new[0][0]]
                    lemmas_sent.append(processed_token)
                    print(tag[0], tag[1], pro_token_new, processed_token)
                else:
                    lemmas_sent.append(pro_token)
            #print(tag[0], lemmas_sent[-1])

    return ' '.join(lemmas_sent)

def preproces(utterance):
    tokens = word_tokenize(utterance)  # 分词
    tagged_sent = pos_tag(tokens)  # 获取单词词性

    wnl = WordNetLemmatizer()
    lemmas_sent = []




    for tag in tagged_sent:
        # print(tag)
        if tag[1] == 'CD' and tag[0] not in ['one', 'two', 'three'] and len(tag[0]) >= 5:
            lemmas_sent.append('<|number|>')
            print(tag[0])
        else:
            lemmas_sent.append(tag[0])

    return ' '.join(lemmas_sent)


def preproces_nlu(utterance):
    tokens = word_tokenize(utterance)  # 分词
    tagged_sent = pos_tag(tokens)  # 获取单词词性

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        # print(tag)
        lemmas_sent.append(simplemma.lemmatize(tag[0], langdata))
        if tag[0] != lemmas_sent[-1]:
            #print(tag[0], lemmas_sent[-1])
            pass

    return ' '.join(lemmas_sent)
def preproces_new(utterance):
    tokens = word_tokenize(utterance)  # 分词
    tagged_sent = pos_tag(tokens)  # 获取单词词性

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        # print(tag)
        if tag[1] == 'CD' and tag[0] not in ['one', 'two', 'three'] and len(tag[0]) >= 5:
            lemmas_sent.append('<|number|>')
            print(tag[0])
        else:
            lemmas_sent.append(simplemma.lemmatize(tag[0], langdata))
            print(tag[0], lemmas_sent[-1])

    return ' '.join(lemmas_sent)

def data_belief(file_name, target_name,delex=False):
    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    belief_file = open(target_name, 'w', encoding='utf-8')
    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        previous_belief = ''
        for d in dia:
            belief = ''
            belief_states = d['belief_state']
            for belief_state in belief_states:
                belief += belief_state['act'] + ' ' + ' '.join(belief_state['slots'][0]) + ','  # Convert all beliefs into sequences
            belief = belief[:-1]

            if delex:
                user = preproces_new(d['transcript'].lower()).lower()
                system = preproces_new(d['system_transcript'].lower()).lower()
            else:
                user = d['transcript'].lower()
                system = d['system_transcript'].lower()

            turn_labels = ''
            for turn_label in d['turn_label']:
                turn_labels += turn_label[0]+' '+turn_label[1]+','

            turn_labels = turn_labels[:-1]
            if system != "":
                history += ' <|system|> ' + system + ' <|user|> ' + user
            else:
                history += ' <|user|> ' + user
            belief_file.write('<|endoftext|> <|context|> ' + history +' <|endofcontext|> <|belief|> '+ belief + ' <|endofbelief|> <|endoftext|> \n')
        belief_file.write('\n')
    belief_file.close()


def data_belief_user(file_name, target_name,delex=False):# inputs: user utterance and system_actions
    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    belief_file = open(target_name, 'w', encoding='utf-8')
    slot_value_en = json.load(open('data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))['informable']
    slot_value_de = json.load(open('data/ontology/ontology_dstc2_de.json', 'r', encoding='utf-8'))['informable']

    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        previous_belief = ''
        for d in dia:
            belief = ''
            belief_states = d['belief_state']
            for belief_state in belief_states:
                belief += belief_state['act'] + ' ' + ' '.join(belief_state['slots'][0]) + ','  # Convert all beliefs into sequences
            belief = belief[:-1]

            system_actions = ''
            system_acts = d['system_acts']
            for system_act in system_acts:
                if isinstance(system_act, str):
                    system_actions += 'request slot '+ system_act+','
                else:
                    system_actions +=  'inform ' + system_act[0] +' '+ system_act[1]+','

            system_actions = system_actions[:-1]
            if delex:
                user = preproces_new(d['transcript'].lower()).lower()
            else:
                user = d['transcript'].lower()
            system = d['system_transcript'].lower()
            turn_labels = ''
            for turn_label in d['turn_label']:
                turn_labels += turn_label[0]+' '+turn_label[1]+','
            if system != "":
                history += ' <|system_actions|> ' + system_actions + ' <|user|> ' + user
            else:
                history += ' <|user|> ' + user
            belief_file.write('<|endoftext|> <|context|> ' + history +' <|endofcontext|>  <|belief|> ' + belief + ' <|endofbelief|> <|endoftext|> \n')
        belief_file.write('\n')
    belief_file.close()
def compute_weights(file_name):
    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    weight = [0,0,0,0,0]
    turn = 0
    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        for d in dia:
            belief = ''
            turn+=1
            belief_states = d['belief_state']
            for belief_state in belief_states:
                act = belief_state['act']
                if act == 'request':
                    weight[0]+=1
                else:
                    weight[1]+=1
                    slot = belief_state['slots'][0]
                    if slot[0]=='food':
                        weight[2]+=1
                    elif slot[0]== 'price range':
                        weight[3] += 1
                    else:
                        weight[4] += 1

    print("weights:",weight)
    print("turns:",turn)

def data_belief_turn(file_name, target_name):
    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    belief_file = open(target_name, 'w', encoding='utf-8')
    slot_value_en = json.load(open('data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))['informable']
    slot_value_de = json.load(open('data/ontology/ontology_dstc2_de.json', 'r', encoding='utf-8'))['informable']

    # get all slot-value pairs
    slot_value_pairs = ''
    for slot in slot_value_de.keys():
        values = slot_value_de[slot]
        for value in values:
            slot_value_pairs += slot + ' ' + value + ','

    slot_value_pairs = slot_value_pairs[:-1]
    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        previous_belief = ''
        for d in dia:
            belief = ''
            belief_states = d['belief_state']
            for belief_state in belief_states:
                belief += belief_state['act'] + ' ' + ' '.join(belief_state['slots'][0]) + ','  # Convert all beliefs into sequences
            belief = belief[:-1]

            system_actions = ''
            system_acts = d['system_acts']
            for system_act in system_acts:
                if isinstance(system_act, str):
                    system_actions += 'request '+ system_act+','
                else:
                    system_actions +=  'request ' + system_act[0] +' '+ system_act[1]+','

            system_actions = system_actions[:-1]
            user = d['transcript']
            system = d['system_transcript']
            turn_labels = ''
            for turn_label in d['turn_label']:
                if 'request' in turn_label:
                    turn_labels += turn_label[0]+' '+turn_label[1]+','
                else:
                    turn_labels += 'inform '+turn_label[0] + ' ' + turn_label[1] + ','
            turn_labels = turn_labels[:-1]
            if system != "":
                history += ' <|system|> ' + system + ' <|user|> ' + user
            else:
                history += ' <|user|> ' + user
            belief_file.write('<|endoftext|> <|context|> ' + history +' <|endofcontext|>  <|belief|> ' + turn_labels + ' <|endofbelief|> <|endoftext|> \n')
            previous_belief = belief
        belief_file.write('\n')
    belief_file.close()

def data_belief_match(file_name,target_name, match_language,is_turn_label=False,delex=False):
    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    belief_file = open(target_name, 'w', encoding='utf-8')
    slot_value_en = json.load(open('data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))['informable']
    slot_value_de = json.load(open('data/ontology/ontology_dstc2_de.json', 'r', encoding='utf-8'))['informable']
    slot_match = {"food": 'essen', "price range": 'preisklasse', "area": 'gegend'}
    if match_language == 'de':
        f = open('data/dst_vocab/en2de_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'es ist egal'
    else:
        f = open('data/dst_vocab/en2it_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'non importa'

    wordnet_lemmatizer = WordNetLemmatizer()
    punctuation = [',', '?', '.']
    #de_mapping['price ranges'] = 'preisklasse'
    #de_mapping['price ranges'] = 'prezzo'
    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        previous_belief = ''
        for d in dia:
            belief = ''
            '''belief_states = d['belief_state']
            for belief_state in belief_states:
                belief += belief_state['act'] + ' ' + ' '.join(
                    belief_state['slots'][0]) + ','  # Convert all beliefs into sequences
            belief = belief[:-1]'''

            system_actions = ''
            system_acts = d['system_acts']
            for system_act in system_acts:
                if isinstance(system_act, str):
                    system_actions += 'request ' + system_act + ','
                else:
                    system_actions += 'request ' + system_act[0] + ' ' + system_act[1] + ','

            system_actions = system_actions[:-1]
            if delex:
                user = preproces_new(d['transcript'].lower()).lower()
                system = preproces_new(d['system_transcript'].lower())
            else:
                user = d['transcript'].lower()
                system = d['system_transcript'].lower()


            for punc in punctuation:
                user = user.replace(punc, ' '+punc)
                system = system.replace(punc, ' ' + punc)

            turn_labels = ''
            if is_turn_label:
                for turn_label in d['turn_label']:
                    if 'request' in turn_label:
                        slot = turn_label[1]
                        user = re.sub(slot, de_mapping[slot], user)
                        system = re.sub(slot, de_mapping[slot], system)
                        turn_labels += turn_label[0] + ' ' + de_mapping[slot] + ','

                    else:
                        slot = turn_label[0]
                        value = turn_label[1]
                        user = re.sub(slot, de_mapping[slot], user)
                        system = re.sub(slot, de_mapping[slot], system)
                        if value in de_mapping.keys():
                            user = re.sub(value, de_mapping[value], user)
                            turn_labels += 'inform ' + slot + ' ' + de_mapping[value] + ','
                        else:
                            turn_labels += 'inform ' + slot + ' ' + value + ','
                turn_labels = turn_labels[:-1]

            else:
                for belief_state in d['belief_state']:

                    if belief_state['act'] == 'request':
                        slot =belief_state['slots'][0][1]
                        #print(slot)
                        belief += 'request slot ' + de_mapping[slot] + ','
                    else:

                        slot = belief_state['slots'][0][0]
                        #print(slot)
                        value = belief_state['slots'][0][1]
                        if value in de_mapping.keys():

                            belief += 'inform ' + de_mapping[slot] + ' ' + de_mapping[value] + ','
                        else:
                            belief += 'inform ' + de_mapping[slot] + ' ' + value + ','
                belief = belief[:-1]

            for key, value in de_mapping.items():
                if len(key.split()) > 1:
                    if key == "price range":  ## could be price ranges in the utterance
                        user = user.replace("price ranges", value)
                        system = system.replace("price ranges", value)
                    user = user.replace(key, value)
                    system = system.replace(key, value)
                else:
                    splits_user = user.split()
                    for i, word in enumerate(splits_user):
                        if word == key: splits_user[i] = value
                    user = " ".join(splits_user)

                    splits_system = system.split()
                    for i, word in enumerate(splits_system):
                        if word == key: splits_system[i] = value
                    system = " ".join(splits_system)

            if system != "":
                history += ' <|system|> ' + system  + ' <|user|> ' + user
            else:
                history += ' <|user|> ' + user



            belief_file.write(
                '<|endoftext|> <|context|> ' + history.lower() + ' <|endofcontext|>  <|belief|> ' + belief + ' <|endofbelief|> <|endoftext|> \n')
            previous_belief = belief
        belief_file.write('\n')
    belief_file.close()


def data_belief_user_match(file_name,target_name, match_language,is_turn_label=False,delex=False ):
    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    belief_file = open(target_name, 'w', encoding='utf-8')
    slot_value_en = json.load(open('data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))['informable']
    slot_value_de = json.load(open('data/ontology/ontology_dstc2_de.json', 'r', encoding='utf-8'))['informable']
    slot_match = {"food": 'essen', "price range": 'preisklasse', "area": 'gegend'}
    if match_language == 'de':
        f = open('data/dst_vocab/en2de_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'es ist egal'
    else:
        f = open('data/dst_vocab/en2it_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'non importa'

    wordnet_lemmatizer = WordNetLemmatizer()
    punctuation = [',', '?', '.']
    #de_mapping['price ranges'] = 'preisklasse'
    #de_mapping['price ranges'] = 'prezzo'
    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        previous_belief = ''
        for d in dia:
            belief = ''
            '''belief_states = d['belief_state']
            for belief_state in belief_states:
                belief += belief_state['act'] + ' ' + ' '.join(
                    belief_state['slots'][0]) + ','  # Convert all beliefs into sequences
            belief = belief[:-1]'''

            system_actions = ''
            system_acts = d['system_acts']
            for system_act in system_acts:
                if isinstance(system_act, str):
                    system_actions += 'request slot ' + system_act + ','
                else:
                    system_actions += 'inform ' + system_act[0] + ' ' + system_act[1] + ','


            system_actions = system_actions[:-1].lower()
            if delex:
                user = preproces_new(d['transcript'].lower())
            else:
                user = d['transcript'].lower()
            system = d['system_transcript'].lower()



            for punc in punctuation:
                user = user.replace(punc, ' '+punc)
                #system = system.replace(punc, ' ' + punc)

            turn_labels = ''
            if is_turn_label:
                for turn_label in d['turn_label']:
                    if 'request' in turn_label:
                        slot = turn_label[1]
                        user = re.sub(slot, de_mapping[slot], user)
                        system = re.sub(slot, de_mapping[slot], system)
                        turn_labels += turn_label[0] + ' ' + de_mapping[slot] + ','

                    else:
                        slot = turn_label[0]
                        value = turn_label[1]
                        user = re.sub(slot, de_mapping[slot], user)
                        system = re.sub(slot, de_mapping[slot], system)
                        if value in de_mapping.keys():
                            user = re.sub(value, de_mapping[value], user)
                            turn_labels += 'inform ' + slot + ' ' + de_mapping[value] + ','
                        else:
                            turn_labels += 'inform ' + slot + ' ' + value + ','
                turn_labels = turn_labels[:-1]

            else:
                for belief_state in d['belief_state']:

                    if belief_state['act'] == 'request':
                        slot =belief_state['slots'][0][1]
                        #print(slot)
                        belief += 'request slot ' + de_mapping[slot] + ','
                    else:

                        slot = belief_state['slots'][0][0]
                        #print(slot)
                        value = belief_state['slots'][0][1]
                        if value in de_mapping.keys():

                            belief += 'inform ' + de_mapping[slot] + ' ' + de_mapping[value] + ','
                        else:
                            belief += 'inform ' + de_mapping[slot] + ' ' + value + ','
                belief = belief[:-1]

            for key, value in de_mapping.items():
                if len(key.split()) > 1:
                    if key == "price range":  ## could be price ranges in the utterance
                        user = user.replace("price ranges", value)
                        system_actions = system_actions.replace("price ranges", value)
                    user = user.replace(key, value)
                    system_actions = system_actions.replace(key, value)
                else:
                    splits_user = user.split()
                    for i, word in enumerate(splits_user):
                        if word == key: splits_user[i] = value
                    user = " ".join(splits_user)

                    splits_system_actions = system_actions.split()
                    for i, word in enumerate(splits_system_actions):
                        if word == key: splits_system_actions[i] = value
                    system_actions = " ".join(splits_system_actions)

            if system != "":
                history += ' <|system_actions|> ' + system_actions  + ' <|user|> ' + user
            else:
                history += ' <|user|> ' + user



            belief_file.write(
                '<|endoftext|> <|context|> ' + history + ' <|endofcontext|>  <|belief|> ' + belief + ' <|endofbelief|> <|endoftext|> \n')
        belief_file.write('\n')
    belief_file.close()

def data_belief_user_match_moretokens(file_name,target_name, match_language,is_turn_label=False,delex=False ):
    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    belief_file = open(target_name, 'w', encoding='utf-8')
    slot_value_en = json.load(open('data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))['informable']
    slot_value_de = json.load(open('data/ontology/ontology_dstc2_de.json', 'r', encoding='utf-8'))['informable']
    slot_match = {"food": 'essen', "price range": 'preisklasse', "area": 'gegend'}
    match_dict = get_dict(match_language)

    if match_language == 'de':
        f = open('data/dst_vocab/en2de_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'es ist egal'
    else:
        f = open('data/dst_vocab/en2it_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'non importa'

    wordnet_lemmatizer = WordNetLemmatizer()
    punctuation = [',', '?', '.']
    #de_mapping['price ranges'] = 'preisklasse'
    #de_mapping['price ranges'] = 'prezzo'
    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        previous_belief = ''
        for d in dia:
            belief = ''
            '''belief_states = d['belief_state']
            for belief_state in belief_states:
                belief += belief_state['act'] + ' ' + ' '.join(
                    belief_state['slots'][0]) + ','  # Convert all beliefs into sequences
            belief = belief[:-1]'''

            system_actions = ''
            system_acts = d['system_acts']
            for system_act in system_acts:
                if isinstance(system_act, str):
                    system_actions += 'request slot ' + system_act + ','
                else:
                    system_actions += 'inform ' + system_act[0] + ' ' + system_act[1] + ','


            system_actions = system_actions[:-1].lower()
            if delex:
                user = preproces_more_tokens(d['transcript'].lower(), match_dict)
            else:
                user = d['transcript'].lower()
            system = d['system_transcript'].lower()



            for punc in punctuation:
                user = user.replace(punc, ' '+punc)
                #system = system.replace(punc, ' ' + punc)

            turn_labels = ''
            if is_turn_label:
                for turn_label in d['turn_label']:
                    if 'request' in turn_label:
                        slot = turn_label[1]
                        user = re.sub(slot, de_mapping[slot], user)
                        system = re.sub(slot, de_mapping[slot], system)
                        turn_labels += turn_label[0] + ' ' + de_mapping[slot] + ','

                    else:
                        slot = turn_label[0]
                        value = turn_label[1]
                        user = re.sub(slot, de_mapping[slot], user)
                        system = re.sub(slot, de_mapping[slot], system)
                        if value in de_mapping.keys():
                            user = re.sub(value, de_mapping[value], user)
                            turn_labels += 'inform ' + slot + ' ' + de_mapping[value] + ','
                        else:
                            turn_labels += 'inform ' + slot + ' ' + value + ','
                turn_labels = turn_labels[:-1]

            else:
                for belief_state in d['belief_state']:

                    if belief_state['act'] == 'request':
                        slot =belief_state['slots'][0][1]
                        #print(slot)
                        belief += 'request slot ' + de_mapping[slot] + ','
                    else:

                        slot = belief_state['slots'][0][0]
                        #print(slot)
                        value = belief_state['slots'][0][1]
                        if value in de_mapping.keys():

                            belief += 'inform ' + de_mapping[slot] + ' ' + de_mapping[value] + ','
                        else:
                            belief += 'inform ' + de_mapping[slot] + ' ' + value + ','
                belief = belief[:-1]

            for key, value in de_mapping.items():
                if len(key.split()) > 1:
                    if key == "price range":  ## could be price ranges in the utterance
                        user = user.replace("price ranges", value)
                        system_actions = system_actions.replace("price ranges", value)
                    user = user.replace(key, value)
                    system_actions = system_actions.replace(key, value)
                else:
                    splits_user = user.split()
                    for i, word in enumerate(splits_user):
                        if word == key: splits_user[i] = value
                    user = " ".join(splits_user)

                    splits_system_actions = system_actions.split()
                    for i, word in enumerate(splits_system_actions):
                        if word == key: splits_system_actions[i] = value
                    system_actions = " ".join(splits_system_actions)

            if system != "":
                history += ' <|system_actions|> ' + system_actions  + ' <|user|> ' + user
            else:
                history += ' <|user|> ' + user



            belief_file.write(
                '<|endoftext|> <|context|> ' + history + ' <|endofcontext|>  <|belief|> ' + belief + ' <|endofbelief|> <|endoftext|> \n')
        belief_file.write('\n')
    belief_file.close()


def data_belief_user_COSDA(file_name, target_name, src_lang, match_language, is_turn_label=False, delex=False,
                           split_lang=0.5, seed=1234):
    random.seed(seed)
    def cross(worddict, x):
        lang_pro = random.random()
        if lang_pro > (1-split_lang):
            lan = 1
        else:
            lan = 0
        # lan = random.randint(0, 1)
        if x in worddict[lan] and 1 >= random.random():
            return worddict[lan][x][random.randint(0, len(worddict[lan][x]) - 1)]
        else:
            return x

    def cross_str(x):
        raw = x.lower().split(" ")
        out = ""
        for xx in raw:
            out += cross(worddict, xx)
            out += " "
        return out


    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    belief_file = open(target_name, 'w', encoding='utf-8')
    slot_value_en = json.load(open('data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))['informable']
    slot_value_de = json.load(open('data/ontology/ontology_dstc2_de.json', 'r', encoding='utf-8'))['informable']
    slot_match = {"food": 'essen', "price range": 'preisklasse', "area": 'gegend'}
    match_dict_de = get_dict(src_lang, match_language)
    match_dict_it = get_dict(match_language, src_lang)
    worddict = []
    worddict.append(match_dict_de)
    worddict.append(match_dict_it)

    if match_language == 'de':
        f = open('data/dst_vocab/en2de_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'es ist egal'

        # it->en->de
        f = open('data/dst_vocab/en2it_onto_for_mix.dict', 'rb')
        de_mapping_it = pickle.load(f)
        for key, value in de_mapping_it.items():
            de_mapping[value] = de_mapping[key]

        de_mapping['non importa'] = 'es ist egal'
    elif match_language == 'it':
        f = open('data/dst_vocab/en2it_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'non importa'

        # de->en->it
        f = open('data/dst_vocab/en2de_onto_for_mix.dict', 'rb')
        de_mapping_de = pickle.load(f)
        for key, value in de_mapping_de.items():
            de_mapping[value] = de_mapping[key]

        de_mapping['es ist egal'] = 'non importa'
    elif match_language == 'en':
        de_mapping = {}
        f = open('data/dst_vocab/en2de_onto_for_mix.dict', 'rb')
        de_mapping_de = pickle.load(f)
        for key, value in de_mapping_de.items():
            de_mapping[value] = key

        f = open('data/dst_vocab/en2it_onto_for_mix.dict', 'rb')
        de_mapping_it = pickle.load(f)
        for key, value in de_mapping_it.items():
            de_mapping[value] = key

        de_mapping['es ist egal'] = 'dontcare'
        de_mapping['non importa'] = 'dontcare'

    wordnet_lemmatizer = WordNetLemmatizer()
    punctuation = [',', '?', '.']
    #de_mapping['price ranges'] = 'preisklasse'
    #de_mapping['price ranges'] = 'prezzo'
    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        previous_belief = ''
        for d in dia:
            belief = ''
            '''belief_states = d['belief_state']
            for belief_state in belief_states:
                belief += belief_state['act'] + ' ' + ' '.join(
                    belief_state['slots'][0]) + ','  # Convert all beliefs into sequences
            belief = belief[:-1]'''

            system_actions = ''
            system_acts = d['system_acts']
            for system_act in system_acts:
                if isinstance(system_act, str):
                    system_actions += 'request slot ' + system_act + ','
                else:
                    system_actions += 'inform ' + system_act[0] + ' ' + system_act[1] + ','


            system_actions = system_actions[:-1].lower()
            if delex:
                user = cross_str(d['transcript'].lower())
            else:
                user = d['transcript'].lower()
            system = d['system_transcript'].lower()



            for punc in punctuation:
                user = user.replace(punc, ' '+punc)
                #system = system.replace(punc, ' ' + punc)

            turn_labels = ''
            if is_turn_label:
                for turn_label in d['turn_label']:
                    if 'request' in turn_label:
                        slot = turn_label[1]
                        user = re.sub(slot, de_mapping[slot], user)
                        system = re.sub(slot, de_mapping[slot], system)
                        turn_labels += turn_label[0] + ' ' + de_mapping[slot] + ','

                    else:
                        slot = turn_label[0]
                        value = turn_label[1]
                        user = re.sub(slot, de_mapping[slot], user)
                        system = re.sub(slot, de_mapping[slot], system)
                        if value in de_mapping.keys():
                            user = re.sub(value, de_mapping[value], user)
                            turn_labels += 'inform ' + slot + ' ' + de_mapping[value] + ','
                        else:
                            turn_labels += 'inform ' + slot + ' ' + value + ','
                turn_labels = turn_labels[:-1]

            else:
                for belief_state in d['belief_state']:

                    if belief_state['act'] == 'request':
                        slot =belief_state['slots'][0][1]
                        #print(slot)
                        if slot in de_mapping.keys():
                            belief += 'request slot ' + de_mapping[slot] + ','
                    else:

                        slot = belief_state['slots'][0][0]
                        #print(slot)
                        value = belief_state['slots'][0][1]
                        if slot in de_mapping.keys() and value in de_mapping.keys():

                            belief += 'inform ' + de_mapping[slot] + ' ' + de_mapping[value] + ','
                        elif slot in de_mapping.keys():
                            belief += 'inform ' + de_mapping[slot] + ' ' + value + ','
                belief = belief[:-1]

            for key, value in de_mapping.items():
                if len(key.split()) > 1:
                    if key == "price range":  ## could be price ranges in the utterance
                        user = user.replace("price ranges", value)
                        system_actions = system_actions.replace("price ranges", value)
                    user = user.replace(key, value)
                    system_actions = system_actions.replace(key, value)
                else:
                    splits_user = user.split()
                    for i, word in enumerate(splits_user):
                        if word == key: splits_user[i] = value
                    user = " ".join(splits_user)

                    splits_system_actions = system_actions.split()
                    for i, word in enumerate(splits_system_actions):
                        if word == key: splits_system_actions[i] = value
                    system_actions = " ".join(splits_system_actions)

            if system != "":
                history += ' <|system_actions|> ' + system_actions  + ' <|user|> ' + user
            else:
                history += ' <|user|> ' + user



            belief_file.write(
                '<|endoftext|> <|context|> ' + history + ' <|endofcontext|>  <|belief|> ' + belief + ' <|endofbelief|> <|endoftext|> \n')
        belief_file.write('\n')
    belief_file.close()

def dst_data_count(file_name,target_name ):
    dialogues = json.load(open(file_name, 'r', encoding='utf-8'))
    # belief_file = open(target_name, 'w', encoding='utf-8')
    # slot_value_en = json.load(open('data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))['informable']
    # slot_value_de = json.load(open('data/ontology/ontology_dstc2_de.json', 'r', encoding='utf-8'))['informable']
    # slot_match = {"food": 'essen', "price range": 'preisklasse', "area": 'gegend'}
    # if match_language == 'de':
    #     f = open('data/dst_vocab/en2de_onto_for_mix.dict', 'rb')
    #     de_mapping = pickle.load(f)
    #     de_mapping['dontcare'] = 'es ist egal'
    # else:
    #     f = open('data/dst_vocab/en2it_onto_for_mix.dict', 'rb')
    #     de_mapping = pickle.load(f)
    #     de_mapping['dontcare'] = 'non importa'
    #
    # wordnet_lemmatizer = WordNetLemmatizer()
    # punctuation = [',', '?', '.']
    #de_mapping['price ranges'] = 'preisklasse'
    #de_mapping['price ranges'] = 'prezzo'
    count = 0
    request_none = 0
    slot_none = 0
    for dialogue in dialogues:
        history = ''
        dia = dialogue['dialogue']
        previous_belief = ''
        for d in dia:
            belief = ''
            belief_states = d['belief_state']
            belief_request = True
            belief_slot = True
            for belief_state in belief_states:
                if belief_state['act'] == 'request':
                    belief_request = False
            turn_slots = d['turn_label']
            for turn_slot in turn_slots:
                if turn_slot[0] in ['area', 'food', 'price range']:
                    belief_slot = False
            if belief_request:
                request_none += 1
            if belief_slot:
                slot_none += 1
            count += 1
    print(count, request_none, slot_none, request_none/count, slot_none/count)




def mask():
    en_ontology = json.load(open('data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))['informable']
    slot_value = set()
    for x in en_ontology.keys():
        slot_value |= set(en_ontology[x])
    result = json.load(open('MT5_mapping_hidden_ende_en_generate_avg_acc.json', 'r', encoding='utf-8'))
    count = 0
    for dialogue in result:
        dia = dialogue['dialogue']
        for turn in dia:
            Prediction = turn[2]['Prediction']
            for key in Prediction.keys():
                if key == 'request':
                  for x in Prediction[key]:
                      count += 1
                      if x not in slot_value:
                          print(x)

                else:
                    predict = Prediction[key]
                    count += 1
                    if predict != 'none' and predict != 'dontcare' and predict not in slot_value:
                        print(predict)
    print(count)


def parse_tsv(data_path, intent_set=[], slot_set=["O"], istrain=True):
    """
    Input:
        data_path: the path of data
        intent_set: set of intent (empty if it is train data)
        slot_set: set of slot type (empty if it is train data)
    Output:
        data_tsv: {"text": [[token1, token2, ...], ...], "slot": [[slot_type1, slot_type2, ...], ...], "intent": [intent_type, ...]}
        intent_set: set of intent
        slot_set: set of slot type
    """
    slot_type_list = ["alarm", "datetime", "location", "reminder", "weather"]
    data_tsv = {"text": [], "slot": [], "intent": []}
    with open(data_path, encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            intent = line[0]
            if istrain == True and intent not in intent_set: intent_set.append(intent)
            if istrain == False and intent not in intent_set:
                intent_set.append(intent)
                # logger.info("Found intent %s not in train data" % intent)
                # print("Found intent %s not in train data" % intent)
            slot_splits = line[1].split(",")
            slot_line = []
            slot_flag = True
            if line[1] != '':
                for item in slot_splits:
                    item_splits = item.split(":")
                    assert len(item_splits) == 3
                    # slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2].split("/")[0]}
                    slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2]}
                    flag = False
                    for slot_type in slot_type_list:
                        if slot_type in slot_item["slot"]:
                            flag = True

                    if flag == False:
                        slot_flag = False
                        break
                    # if istrain == True and slot_item["slot"] not in slot_set: slot_set.append(slot_item["slot"])
                    # if istrain == False and slot_item["slot"] not in slot_set:
                    #     slot_set.append(slot_item["slot"])
                    #     # logger.info("Found slot %s not in train data" % item_splits[2])
                    #     # print("Found slot %s not in train data" % item_splits[2])
                    slot_line.append(slot_item)

            if slot_flag == False:
                # slot flag not correct
                continue

            token_part = json.loads(line[4])
            tokens = token_part["tokenizations"][0]["tokens"]
            tokenSpans = token_part["tokenizations"][0]["tokenSpans"]

            data_tsv["text"].append(tokens)
            data_tsv["intent"].append(intent)
            slots = []
            for tokenspan in tokenSpans:
                nolabel = True
                for slot_item in slot_line:
                    start = tokenspan["start"]
                    # if int(start) >= int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                    if int(start) == int(slot_item["start"]):
                        nolabel = False
                        slot_ = "B-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                    if int(start) > int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                        nolabel = False
                        slot_ = "I-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                if nolabel == True: slots.append("O")
            data_tsv["slot"].append(slots)

            assert len(slots) == len(tokens)

    return data_tsv, intent_set, slot_set


def gen_mix_lang_data(data, token_mapping):
    data_new = {"text": [], "slot": [], "intent": []}
    data_new["slot"] = data["slot"]
    data_new["intent"] = data["intent"]
    for token_list in data["text"]:
        token_list_new = []
        for token in token_list:
            if token in token_mapping:
                token = token_mapping[token]
            token_list_new.append(token)

        assert len(token_list_new) == len(token_list)
        data_new["text"].append(token_list_new)

    return data_new

def nlu(file_name, target_name):

    import scipy as sp
    result = set()
    save_file = open(target_name, 'w', encoding='utf-8')
    #save_file = sp.genfromtxt("data/nlu/en/train-en.tsv", delimiter="\t")
    #save_file = pd.read_csv('data/nlu/en/train-en.tsv',sep = '\t',encoding='utf-8')
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        #reader = sp.genfromtxt("data/nlu/en/train-en.tsv", delimiter="\t")
        for row in reader:
            #print(row[1])
            if row != '' and row != '\n':
                slot_values_raw = row[1].split(',')
                slot_values = ''
                if slot_values_raw != '' and row != '\n':
                    for slot_value in slot_values_raw:
                        if slot_value != '':
                            start_id = int(slot_value.split(':')[0])
                            end_id = int(slot_value.split(':')[1])
                            value = row[2][start_id:end_id]
                            slot = slot_value.split(':')[-1]
                            slot_values += slot + ' ' + value + ','
                    slot_values = slot_values[:-1]
                    temp = '<|endoftext|> <|context|> '+ row[2].lower() + '<|endofcontext|> '+\
                            '<|intent_slot|> <|intent|> '+ row[0] +' <|slot_value|> '+ slot_values\
                            + '<|endof_intent_slot|> <|endoftext|>'
                    result.add(temp)
    for x in result:
        save_file.write(x + '\n')

    save_file.close()

def nlu_bio(file_name, target_name):

    import scipy as sp
    result = set()
    save_file = open(target_name, 'w', encoding='utf-8')
    #save_file = sp.genfromtxt("data/nlu/en/train-en.tsv", delimiter="\t")
    #save_file = pd.read_csv('data/nlu/en/train-en.tsv',sep = '\t',encoding='utf-8')
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        #reader = sp.genfromtxt("data/nlu/en/train-en.tsv", delimiter="\t")
        for row in reader:
            #print(row[1])
            if row != '' and row != '\n':
                slot_values_raw = row[1].split(',')
                slot_values = ''
                tag = {}
                res = ''
                if slot_values_raw != '' and row != '\n':
                    for slot_value in slot_values_raw:
                        if slot_value != '':
                            start_id = int(slot_value.split(':')[0])
                            end_id = int(slot_value.split(':')[1])
                            value = row[2][start_id:end_id]
                            slot = slot_value.split(':')[-1]
                            slot_values += slot + ' ' + value + ','
                            for index,token in enumerate(value.strip().split()):
                                if index == 0:
                                     tag[token] = 'B-' + slot
                                else:
                                     tag[token] = 'I-' + slot
                for x in [',', '.', '?']:  #Split last token and punctuation.
                    row[2] = row[2].replace(x, ' '+x)
                for token in row[2].split():
                    if token in tag.keys():
                        res += tag[token] + ' '
                    else:
                        res += 'O '
                temp = '<|endoftext|> <|context|> ' + preproces_nlu(row[2].lower()) + '<|endofcontext|> ' + \
                       '<|intent_slot|> <|intent|> ' + row[0] + ' <|slot_value|> ' + res \
                       + '<|endof_intent_slot|> <|endoftext|>'
                result.add(temp)
    for x in sorted(result):
        save_file.write(x + '\n')
    save_file.write('\n')
    save_file.close()

def nlu_bio_match(file_name, target_name,match_language):

    if match_language == 'es':
        f = open('data/dst_vocab/en2es_20.attn.dict', 'rb')
        de_mapping = pickle.load(f)
    if match_language == 'th':
        f = open('data/dst_vocab/en2th_20.attn.dict', 'rb')
        de_mapping = pickle.load(f)

    import scipy as sp
    result = set()
    save_file = open(target_name, 'w', encoding='utf-8')
    count = 0
    repeat =[]
    #save_file = sp.genfromtxt("data/nlu/en/train-en.tsv", delimiter="\t")
    #save_file = pd.read_csv('data/nlu/en/train-en.tsv',sep = '\t',encoding='utf-8')
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        #reader = sp.genfromtxt("data/nlu/en/train-en.tsv", delimiter="\t")
        for row in reader:
            #print(row[1])
            if row != '' and row != '\n':
                slot_values_raw = row[1].split(',')
                slot_values = ''
                tag = {}
                res = ''
                if slot_values_raw != '' and row != '\n':
                    for slot_value in slot_values_raw:
                        if slot_value != '':
                            start_id = int(slot_value.split(':')[0])
                            end_id = int(slot_value.split(':')[1])
                            value = row[2][start_id:end_id]
                            slot = slot_value.split(':')[-1]
                            slot_values += slot + ' ' + value + ','
                            for index,token in enumerate(value.strip().split()):
                                if index == 0:
                                     tag[token] = 'B-' + slot
                                else:
                                     tag[token] = 'I-' + slot
                for x in [',', '.', '?']:  #Split last token and punctuation.
                    row[2] = row[2].replace(x, ' '+x)
                for token in row[2].split():
                    if token in tag.keys():
                        res += tag[token] + ' '
                    else:
                        res += 'O '

                user = copy.deepcopy(row[2].lower())
                for key, value in de_mapping.items():
                    if len(key.split()) > 1:
                        user = user.replace(key, value)
                    else:
                        splits_user = user.split()
                        for i, word in enumerate(splits_user):
                            if word == key: splits_user[i] = value
                        user = " ".join(splits_user)


                temp = '<|endoftext|> <|context|> ' + user + '<|endofcontext|> ' + \
                       '<|intent_slot|> <|intent|> ' + row[0] + ' <|slot_value|> ' + res \
                       + '<|endof_intent_slot|> <|endoftext|>'
                previous_len = len(result)
                result.add(temp)
                if len(result) == previous_len:
                    repeat.append(count)
                count += 1
    for x in result:
        save_file.write(x + '\n')
    print(count)
    save_file.close()
    return repeat

def nlu_bio_CLCSA(file_name, target_name, lang_split, seed=1234):
    random.seed(seed)
    def cross(worddict, x):
        lang_pro = random.random()
        if lang_pro > (1-lang_split):
            lan = 1
        else:
            lan = 0
        #lan = random.randint(0, 1)
        if x in worddict[lan] and 0.9 >= random.random():
            return worddict[lan][x][random.randint(0, len(worddict[lan][x]) - 1)]
        else:
            return x

    def cross_str(x):
        raw = x.lower().split(" ")
        out = ""
        for xx in raw:
            out += cross(worddict, xx)
            out += " "
        return out

    match_dict_es = get_dict('es', 'th')
    match_dict_th = get_dict('th', 'es')
    worddict = []
    worddict.append(match_dict_es)
    worddict.append(match_dict_th)

    save_file = open(target_name, 'w', encoding='utf-8')
    data_train, intent_set, slot_set = parse_tsv(file_name)
    for i in range(len(data_train['text'])):
        temp = '<|endoftext|> <|context|> ' + cross_str(' '.join(data_train['text'][i]).lower()) + '<|endofcontext|> ' + \
                '<|intent_slot|> <|intent|> ' + data_train['intent'][i] + ' <|slot_value|> ' \
               + ' '.join(data_train['slot'][i]) + '<|endof_intent_slot|> <|endoftext|>'
        save_file.write(temp+'\n')
    save_file.close()

def nlu_bio_translate(file_name, target_name):
    trans = translate.Translate()
    save_file = open(target_name, 'a', encoding='utf-8')
    data_train, intent_set, slot_set = parse_tsv(file_name)
    for i in range(24670, len(data_train['text'])):
        trans_text = trans.translate(' '.join(data_train['text'][i]))
        if trans_text:
            trans_text = trans_text.lower()
            temp = '<|endoftext|> <|context|> ' + ' '.join(data_train['text'][i]) + ' <|trans|> ' +trans_text + '<|endofcontext|> ' + \
                   '<|intent_slot|> <|intent|> ' + data_train['intent'][i] + ' <|slot_value|> ' \
                    + ' '.join(data_train['slot'][i]) + '<|endof_intent_slot|> <|endoftext|>'
            print(i)
            save_file.write(temp + '\n')
        else:
            continue
    save_file.close()

def main():
    #lang = 'de'
    for lang in ['en','de','it']:
        for model in ['train','val','test']:
            if model == 'val':
                file_name = 'data/mulwoz/woz_validate_%s.json' %lang
            else:
                file_name = 'data/mulwoz/woz_%s_%s.json' %(model,lang)
            target_name = 'data/mulwoz_process/beliefinput1_%s_%s.txt' %(lang,model)
            data_belief(file_name, target_name, delex=False)
 
def one_hot(tokenizer, vocab_size):
    vocab = {}
    language = ['en', 'de', 'it']
    for x in language:
        temp = 'ontology_dstc2_%s.json' %x
        ontology_temp = json.load(open('../data/ontology/'+temp, 'r', encoding='utf-8'))
        for key, values in ontology_temp['informable'].items():
            vocab[key] = tokenizer.encode(key)[:-1]
            for value in values:
                vocab[value] = tokenizer.encode(value)[:-1]

    vocab['slot'] = tokenizer.encode('slot')[:-1]
    vocab[','] = tokenizer.encode(',')[-1]
    print(len(vocab),vocab)

    result = np.zeros(1, vocab_size)
    for key in vocab.keys():
        for tokens in vocab[key]:
            result[0][tokens] = 1.0
            print("posi:", tokens)
    return result


def upper_bound(evaluated_dialogues,language):
    request = []
    joint = []
    en_slot = ['food', 'price range', 'area']
    de_slot = ['essen', 'preisklasse', 'gegend']
    it_slot = ['cibo', 'prezzo', 'area']

    request_0 = []
    joint_0 = []
    request_1 = []
    joint_1 = []
    eval_numbers = len(evaluated_dialogues)
    if language == 'en':
        slot_dict = en_slot
    elif language == 'de':
        slot_dict = de_slot
    else:
        slot_dict = it_slot
    for index1 in range(len(evaluated_dialogues[0])):
        dia = []
        for x in range(eval_numbers):
            dia.append(evaluated_dialogues[x][index1]['dialogue'])
        for index2 in range(len(dia[0])):
            True_state = dia[0][index2][1]['True State']

            Prediction = []
            request_pre = []
            joint_gold = []
            joint_pre = []

            for y in range(eval_numbers):
                Prediction.append(dia[y][index2][2]['Prediction'])
                request_pre.append(set(Prediction[y]['request']))
                joint_temp = []

                for slot in slot_dict:
                    joint_temp.append(Prediction[y][slot])
                joint_pre.append(joint_temp)
            request_gold = set(True_state['request'])
            #print(Prediction[1])
            for slot in slot_dict:
                joint_gold.append(True_state[slot])

            request_tag =False
            joint_tag =False
            request_0.append(request_gold == request_pre[0])
            request_1.append(request_gold == request_pre[1])
            joint_0.append(joint_gold == joint_pre[0])
            joint_1.append(joint_gold == joint_pre[1])
            for y in range(eval_numbers):
                if request_gold == request_pre[y] and not request_tag:
                    #request.append(True)
                    request_tag = True
                if joint_gold == joint_pre[y] and not joint_tag:
                    #joint.append(True)
                    joint_tag = True

            request.append(request_tag)
            joint.append(joint_tag)


    print("request:", np.mean(request), "req_0:", np.mean(request_0), "req_1:", np.mean(request_1))
    print('joint:', np.mean(joint), 'joint_0:', np.mean(joint_0), 'joint_1:', np.mean(joint_1))

def upper_bound_nlu(evaluated_dialogues):
    index2slot = {'O': 0, 'B-weather/noun': 1, 'I-weather/noun': 2, 'B-location': 3, 'I-location': 4, 'B-datetime': 5,
                  'I-datetime': 6,
                  'B-weather/attribute': 7, 'I-weather/attribute': 8, 'B-reminder/todo': 9, 'I-reminder/todo': 10,
                  'B-alarm/alarm_modifier': 11, 'B-reminder/noun': 12, 'B-reminder/recurring_period': 13,
                  'I-reminder/recurring_period': 14, 'B-reminder/reference': 15, 'I-reminder/noun': 16,
                  'B-reminder/reminder_modifier': 17, 'I-reminder/reference': 18, 'I-reminder/reminder_modifier': 19,
                  'B-weather/temperatureUnit': 20, 'I-alarm/alarm_modifier': 21, 'B-alarm/recurring_period': 22,
                  'I-alarm/recurring_period': 23}

    intent = []
    slot_pre = []
    slot_gold = []
    intent_0 = []
    slot_0 = []
    intent_1 = []
    slot_1 = []
    eval_numbers = len(evaluated_dialogues)

    for index1 in range(len(evaluated_dialogues[0])):
        True_state = evaluated_dialogues[0][index1][1]['True State']
        intent_gold = True_state['intent']
        slot_gold_temp = True_state['slot'].split()
        slot_gold_temp_id = []
        for slot_value in slot_gold_temp:
            if slot_value in index2slot.keys():
                slot_gold_temp_id.append(index2slot[slot_value])
            else:
                slot_gold_temp_id.append(len(index2slot))


        intent_pre = []

        f1_max = 0
        slot_pre_best = []
        slot_gold_new = []
        for x in range(eval_numbers):
            intent_pre.append(evaluated_dialogues[x][index1][2]['Prediction']['intent'])
            slot_temp = evaluated_dialogues[x][index1][2]['Prediction']['slot'].split()

            slot_pre_temp_id = []  # convert slot_values to id
            for slot_value in slot_temp:
                if slot_value in index2slot.keys():
                    slot_pre_temp_id.append(index2slot[slot_value])
                else:
                    slot_pre_temp_id.append(len(index2slot))

            slot_gold_temp_id_copy = copy.deepcopy(slot_gold_temp_id)
            if len(slot_gold_temp_id_copy) < len(slot_pre_temp_id):
                slot_gold_temp_id_copy += ([len(index2slot)] * (len(slot_pre_temp_id) - len(slot_gold_temp_id_copy)))
            elif len(slot_gold_temp_id_copy) > len(slot_pre_temp_id):
                slot_pre_temp_id += ([len(index2slot)] * (len(slot_gold_temp_id_copy) - len(slot_pre_temp_id)))
            #print("slot_temp:", slot_pre_temp_id)

            assert len(slot_gold_temp_id_copy) == len(slot_pre_temp_id)
            f1_temp = f1_score(y_true=slot_gold_temp_id_copy, y_pred=slot_pre_temp_id, average='micro')
            if f1_temp > f1_max:
                slot_pre_best = copy.deepcopy(slot_pre_temp_id)
                f1_max = f1_temp
                slot_gold_new = copy.deepcopy(slot_gold_temp_id_copy)
        #print("slot_pre:", slot_pre_best)
        slot_pre += slot_pre_best
        slot_gold += slot_gold_new


        intent_tag = False
        for x in range(eval_numbers):
            if intent_gold == intent_pre[x] and not intent_tag:
                # request.append(True)
                intent_tag = True
        intent.append(intent_tag)

        intent_0.append(intent_pre[0]==intent_gold)
        intent_1.append(intent_pre[1]==intent_gold)
    print(len(slot_pre), len(slot_gold))
    assert len(slot_pre) == len(slot_gold)
    slot_f1 = f1_score(y_true=slot_gold, y_pred=slot_pre, average='micro')

    print("intent:", np.mean(intent), "intent_0:", np.mean(intent_0), "intent_1:", np.mean(intent_1))
    print('slot:', slot_f1)
def nlu_sim():
    # model = MT5ForConditionalGeneration.from_pretrained("../mt5/")
    # tokenizer = MT5Tokenizer.from_pretrained("../mt5/")
    model = MT5ForConditionalGeneration.from_pretrained("../GPT2-chitchat/model/mt5/")
    tokenizer = MT5Tokenizer.from_pretrained("../GPT2-chitchat/model/mt5/")

    #input_ids = tokenizer.encode('I love you!')
    nlu_en_train = open('data/nlu_process/nlubiodelex_en_train.txt', 'r', encoding='utf-8').read()
    nlu_en_data = nlu_en_train.split('\n')[:-2]
    nlu_en_new = []
    for x in nlu_en_data:
        nlu_en_new.append(x.split('<|context|>')[1].split('<|endofcontext|>')[0])

    nlu_es_train = open('data/nlu_process/nlubiodelex_es_train.txt', 'r', encoding='utf-8').read()
    nlu_es_data = nlu_es_train.split('\n')[:-2]
    nlu_es_new = []
    for x in nlu_es_data:
        nlu_es_new.append(x.split('<|context|>')[1].split('<|endofcontext|>')[0])

    input_ids_en = tokenizer(nlu_en_new, return_tensors="pt", padding=True)['input_ids'][:-1]
    input_ids_es = tokenizer(nlu_es_new, return_tensors="pt", padding=True)['input_ids'][:-1]
    nlu_en_parallel = open('data/nlu_process/nlu_en_parallel.txt', 'w', encoding='utf-8')

    embeddings_en = torch.mean(model.get_input_embeddings()(input_ids_en), dim=-2).to('cuda')
    embeddings_es = torch.mean(model.get_input_embeddings()(input_ids_es), dim=-2).to('cuda')

    for x in embeddings_es:
        sim = torch.pairwise_distance(x, embeddings_en)
        min_index = torch.argmin(sim).item()
        nlu_en_parallel.write(nlu_en_data[min_index]+'\n')

    nlu_en_parallel.close()
        #print(min_index)
    #torch.pairwise_distance(embeddings_en, embeddings_en[1])
    #res = torch.matmul(embeddings_en, embeddings_es.t())
    #print(res[0][0])

def transdata():
    train_en = open('data/mulwoz_process/beliefinput2delex_en_train.txt', 'r', encoding='utf-8').read().split('\n\n')[0:-1]
    valid_en = open('data/mulwoz_process/beliefinput2delex_en_val.txt', 'r', encoding='utf-8').read().split('\n\n')[
               0:-1]

    test_en = open('data/mulwoz_process/beliefinput2delex_en_test.txt', 'r', encoding='utf-8').read().split('\n\n')[
               0:-1]
    train_src_en = open('data/mulwoz_process/beliefinput2delex_en_train_src.txt', 'w', encoding='utf-8')
    train_tgt_en = open('data/mulwoz_process/beliefinput2delex_en_train_tgt.txt', 'w', encoding='utf-8')
    test_src_en = open('data/mulwoz_process/beliefinput2delex_en_test_src.txt', 'w', encoding='utf-8')

    val_src_en = open('data/mulwoz_process/beliefinput2delex_en_val_src.txt', 'w', encoding='utf-8')
    val_tgt_en = open('data/mulwoz_process/beliefinput2delex_en_val_tgt.txt', 'w', encoding='utf-8')
    test_tgt_en = open('data/mulwoz_process/beliefinput2delex_en_test_tgt.txt', 'w', encoding='utf-8')



    for datas in train_en:
        turns = datas.split('\n')
        for turn in turns:
            train_src_en.write(turn.split('<|endofcontext|>')[0].split('<|context|>')[1]+'\n')
            train_tgt_en.write(turn.split('<|endofcontext|>')[1].split('<|endoftext|>')[0] + '\n')

    for datas in valid_en:
        turns = datas.split('\n')
        for turn in turns:
            val_src_en.write(turn.split('<|endofcontext|>')[0].split('<|context|>')[1]+'\n')
            val_tgt_en.write(turn.split('<|endofcontext|>')[1].split('<|endoftext|>')[0] + '\n')


    for datas in test_en:
        turns = datas.split('\n')
        for turn in turns:
            test_src_en.write(turn.split('<|endofcontext|>')[0].split('<|context|>')[1]+'\n')
            test_tgt_en.write(turn.split('<|endofcontext|>')[1].split('<|endoftext|>')[0] + '\n')

def evaluate_avg_acc(generate_path, groundtruth_path):
    outputs = open(generate_path, 'r', encoding='utf-8').read().split('\n')
    dialogue_groundtruth = open(groundtruth_path, 'r', encoding='utf-8').read().split('\n')

    request = []
    joint = []

    print(len(outputs), len(dialogue_groundtruth))
    for index in range(len(outputs)-1):
        groundtruth = dialogue_groundtruth[index].split('<|belief|>')[1].split(
            '<|endofbelief|>')[0]
        #print("groundtrhtu:", groundtruth)
        try:
            generation = outputs[index]
            if '<|belief|>' in generation and '<|endofbelief|>' in generation:
                generation = \
                generation.split('<|belief|>')[1].split(
                    '<|endofbelief|>')[0]
            # generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
            else:
                generation = 'error'
            print('generation:', generation)
        except IndexError:
            generation = 'error'
            #print("error")

        request_gold = set()
        request_pred = set()
        joint_gold = {}
        joint_pred = {}
        if generation != 'error' and generation != ' ':
            for x in generation.split(','):
                if len(x.strip().split()) >= 3:
                    act = x.strip().split()[0]
                    if 'price range' in x:
                        slot = 'price range'
                    else:
                        slot = x.strip().split()[1]
                    value = x.split(slot)[1].strip()
                    if act == 'request':
                        request_pred.add(value)
                    else:
                        joint_pred[slot] = value
                else:
                    request_pred = set('1')
                    joint_pred = {'1': 1}
                    break
        else:
            request_pred = set('1')
            joint_pred = {'1':1}

        for x in groundtruth.split(','):
            act = x.strip().split()[0]
            if 'price range' in x:
                slot = 'price range'
            else:
                slot = x.strip().split()[1]
            value = x.split(slot)[1].strip()
            if act == 'request':
                request_gold.add(value)
            else:
                joint_gold[slot] = value

        print("req_pre:", request_pred, "req_gold:", request_gold)
        print("joint_pre:", joint_pred, "joint_gold:", joint_gold)
        request.append(request_gold == request_pred)
        joint.append(joint_gold == joint_pred)


    print("finishing evaluating. request {}, joint {}, avg_acc {}".format(np.mean(request), np.mean(joint),
                 (np.mean(request) + np.mean(joint)) / 2))

    return (np.mean(request)+np.mean(joint))/2

def bootstrap_sampling(dataset, dataset_number):
    #random.seed(12323)
    dataset_leave = copy.deepcopy(dataset)
    dataset = list(set(dataset))[0:int(len(dataset)*0.7)]
    length = len(dataset)
    dataset_new_list = []
    for number in range(dataset_number):
        dataset_new = []
        for x in range(length):
            randomint = random.randint(0, length-1)
            #print("random_int:", randomint)
            dataset_new.append(dataset[randomint])
        dataset_new_list.append(dataset_new)
        #print(dataset_new_list)
        dataset_leave = list(set(dataset_leave) - set(dataset_new))
        #print("data_set_new:", len(set(dataset_new)))
    # for x in dataset_leave:
    #     print(x)
    return dataset_new_list, dataset_leave

# 将同BIOlabel的等同的数量的特殊字符放到输入中起到一个辅助的信号作用
def bio_label_count(file_name, target_name):
    data_train = open(file_name, 'r', encoding='utf-8').read().split('\n\n')[0:-1]
    save_file = open(target_name, 'w', encoding='utf-8')

    for turns in data_train:
        x = turns.split('\n')
        for y in x:
            label_num = len(y.split('<|slot_value|>')[1].split('<|endof_intent_slot|>')[0].strip().split())
            context_new = y.split('<|endofcontext|>')[0] + ' '.join(['<|N|>'] * label_num) +' <|endofcontext|> ' +\
            y.split('<|endofcontext|>')[1]
            save_file.write(context_new + '\n')
    save_file.write('\n')
    save_file.close()

def nlu_label(file_name, target_name):
    data_train = open(file_name, 'r', encoding='utf-8').read().split('\n\n')[0:-1]
    save_file = open(target_name, 'w', encoding='utf-8')

    for turns in data_train:
        x = turns.split('\n')
        for y in x:
            label_num = len(y.split('<|slot_value|>')[1].split('<|endof_intent_slot|>')[0].strip().split())
            context_new = y.split('<|endofcontext|>')[0] + ' '.join(['<|N|>'] * label_num) +' <|endofcontext|> ' +\
            y.split('<|endofcontext|>')[1]
            save_file.write(context_new + '\n')
    save_file.write('\n')
    save_file.close()

def nlu_trans_mlt(file_name, save_name):

    f = open('data/dst_vocab/en2th_20.attn.dict', 'rb')
    save_file = open(save_name, 'w', encoding='utf-8')
    de_mapping = pickle.load(f)
    datas = open(file_name, 'r', encoding='utf-8').read().split('\n\n')[0:-1]
    for data in datas:
        sentences = data.split('\n')
        for sentence in sentences:
            temp = sentence.split('<|trans|>')[0].split('<|context|>')[1].strip()
            mlt = []
            for token in temp.split():
                if token in de_mapping.keys():
                    mlt_token = de_mapping[token]
                    mlt.append(mlt_token)
                else:
                    mlt.append(token)
            new_sentence = sentence.split('<|endofcontext|>')[0] + '<|mlt|>' + ' '.join(mlt) + '<|endofcontext|>' \
                + sentence.split('<|endofcontext|>')[1]
            save_file.write(new_sentence + '\n')
    save_file.write('\n')


if __name__ == '__main__':

    # NLU
    # nlubiomlt
    for lang in ['en','es','th']:
        nlu_bio('data/nlu/'+lang+'/train-'+lang+'.tsv', 'data/nlu_process/nlubiomlt_'+lang+'_train.txt')
        nlu_bio('data/nlu/' + lang + '/eval-' + lang + '.tsv', 'data/nlu_process/nlubiomlt_' + lang + '_val.txt')
        nlu_bio('data/nlu/' + lang + '/test-' + lang + '.tsv', 'data/nlu_process/nlubiomlt_' + lang + '_test.txt')


    # nluclcsa0.7_es_train.txt
    #nlu_bio_CLCSA('data/nlu/en/train-en.tsv', 'data/nlu_process/nluclcsa0.7_es_train.txt', 0.3)
    #nlu_bio_CLCSA('data/nlu/en/train-en.tsv', 'data/nlu_process/nluclcsa0.7_th_train.txt', 0.7)


    # DST
    # beliefinput2delex
    # for lang in ['en', 'de', 'it']:
    #     data_belief_user('data/mulwoz/woz_train_'+lang+'.json', 'data/mulwoz_process/beliefinput2delex_'+lang+'_train.txt')
    #     data_belief_user('data/mulwoz/woz_validate_'+lang+'.json', 'data/mulwoz_process/beliefinput2delex_'+lang+'_val.txt')
    #     data_belief_user('data/mulwoz/woz_test_' + lang + '.json', 'data/mulwoz_process/beliefinput2delex_' + lang + '_test.txt')


    #beliefCOSDA_de_match
    # data_belief_user_COSDA('data/mulwoz/woz_train_de.json', 'data/mulwoz_process/beliefcrossclcsade1_it_match.txt',
    #                                               src_lang='de', match_language='it', delex=True, split_lang=0)


