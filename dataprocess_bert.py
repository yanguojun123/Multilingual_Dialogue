import pandas
import json

def main():

    dialogues = json.load(open('G:\learn\Multilingual_Dialogue\data\mulwoz\woz_test_it.json','r',encoding='utf-8'))
    dialogues_new = {}
    dialogues_new['dialogues'] = []
    for dialogue in dialogues:
        dialogue_new = {}
        dialogue_new['turns'] =[]
        dialogue_new['dialogue_id'] = dialogue['dialogue_idx']
        for turn in dialogue['dialogue']:
            turn_new = {}
            turn_new['system_acts'] = turn['system_acts']
            turn_new['system_transcript'] = turn['system_transcript']
            turn_new['num'] ={}
            turn_new['belief_state'] = turn['belief_state']
            turn_new['turn_id'] = turn['turn_idx']
            turn_new['transcript'] = turn['transcript'][:-1].split()
            if(len(turn['transcript'])>0):
                turn_new['transcript'].append(turn['transcript'][-1])
            turn_new['turn_label'] = turn['turn_label']
            dialogue_new['turns'].append(turn_new)
        dialogues_new['dialogues'].append(dialogue_new)
    json.dump(dialogues_new, open('G:\learn\code\BERT-Dialog-State-Tracking\data\woz_it\\test.json', 'w', encoding='utf-8'))

def ontology():

    # add  de ontology
    # ontologys = json.load(open('G:\learn\Multilingual_Dialogue\data\ontology\ontology_dstc2_it.json', 'r', encoding='utf-8'))
    # ontologys_new = {}
    # ontologys_new['slots'] = ['gegend','essen','preisklasse','request']
    # ontologys_new['values'] = {}
    # ontologys_new['values']['request'] = ontologys['requestable']
    # ontologys_new['values']['gegend'] = ontologys['informable']['gegend']
    # ontologys_new['values']['essen'] = ontologys['informable']['essen']
    # ontologys_new['values']['preisklasse'] = ontologys['informable']['preisklasse']
    # json.dump(ontologys_new,
    #           open('G:\learn\code\BERT-Dialog-State-Tracking\data\woz_it\ontology.json', 'w', encoding='utf-8'))

    # add it ontology
    ontologys = json.load(
        open('G:\learn\Multilingual_Dialogue\data\ontology\ontology_dstc2_it.json', 'r', encoding='utf-8'))
    ontologys_new = {}
    ontologys_new['slots'] = ['cibo', 'prezzo', 'area', 'request']
    ontologys_new['values'] = {}
    ontologys_new['values']['request'] = ontologys['requestable']
    ontologys_new['values']['cibo'] = ontologys['informable']['cibo']
    ontologys_new['values']['prezzo'] = ontologys['informable']['prezzo']
    ontologys_new['values']['area'] = ontologys['informable']['area']
    json.dump(ontologys_new,
              open('G:\learn\code\BERT-Dialog-State-Tracking\data\woz_it\ontology.json', 'w', encoding='utf-8'))


#main()
ontology()
