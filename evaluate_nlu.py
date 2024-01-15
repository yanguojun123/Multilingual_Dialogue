import logging

import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from Multilingual.conll2002_metrics import *


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_language', default='en', type=str, required=False, help='target language')


    parser.add_argument('--test_file_name', default='data/nlu_process/nlubiomlt_en_test.txt', type=str, required=False, help='原始测试语料')
    parser.add_argument('--result_file_name', default='data/nlu_process/nlubiomlt_en_test.txt', type=str, required=False, help='保存生成结果的路径')



    return parser.parse_args()
def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

count = 0
dialogue_all = []
dialogue_dict = {}
dialogue_dict['dialogue_idx'] = count
dialogue_dict['dialogue'] = []


def evaluate_result(outputs, dialogue_groundtruth):

    slot_pre = []
    slot_gold = []
    intent_pred = []
    intent_gold = []

    intent_set = {'weather/find': 0, 'alarm/set_alarm': 1, 'alarm/show_alarms': 2, 'reminder/set_reminder': 3,
                  'alarm/modify_alarm': 4,
                  'weather/checkSunrise': 5, 'weather/checkSunset': 6, 'alarm/snooze_alarm': 7, 'alarm/cancel_alarm': 8,
                  'reminder/show_reminders': 9, 'reminder/cancel_reminder': 10, 'alarm/time_left_on_alarm': 11}


    for index in range(len(outputs)):
        # example: <|intent_slot|> <|intent|> weather/find <|slot_value|> O O O B-weather/noun I-weather/noun O B-location I-location I-location <|endof_intent_slot|>
        groundtruth = dialogue_groundtruth[index].split('<|intent_slot|>')[1].split(
            '<|endof_intent_slot|>')[0]

        # print("groundtrhtu:", groundtruth)
        try:
            generation = outputs[index]
            if '<|intent_slot|>' in generation and '<|endof_intent_slot|>' in generation:
                generation = \
                    generation.split('<|intent_slot|>')[1].split(
                        '<|endof_intent_slot|>')[0]
            # generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
            else:
                generation = 'error'
            # print('generation:', generation)
        except IndexError:
            generation = 'error'
            # print("error")

        # intent_gold = ''
        # intent_pred = ''
        # joint_gold = {}
        # joint_pred = {}
        slot_values_pre = ''

        # example: weather/find, intent_gold: [0,0,1,2]
        intent_gold.append(intent_set[groundtruth.split('<|slot_value|>')[0].split('<|intent|>')[1].strip()])

        # example: [O, O, O, B-weather/noun, I-weather/noun, O, B-location, I-location, I-location]
        slot_values_gold = groundtruth.split('<|slot_value|>')[1].strip().split()
        for slot_value in slot_values_gold:
            # if slot_value in index2slot.keys():
            #     slot_gold.append(index2slot[slot_value])
            # else:
            slot_gold.append(slot_value)

        if generation != 'error' and generation != ' ':
            if '<|slot_value|>' in generation:

                # example: 'weather/find'
                intent_temp = generation.split('<|slot_value|>')[0].split('<|intent|>')[1].strip()
                if intent_temp in intent_set.keys():
                    # example: [0]
                    intent_pred.append(intent_set[intent_temp])
                else:
                    # example: [12]
                    intent_pred.append(len(intent_set))
                slot_values_pre = generation.split('<|slot_value|>')[1].strip().split()
                if len(slot_values_gold) == len(slot_values_pre):
                    for slot_value in slot_values_pre:
                        # if slot_value in index2slot.keys():
                        #     slot_pre.append(index2slot[slot_value])
                        # else:
                        slot_pre.append(slot_value)
                else:
                    if len(slot_values_gold) < len(slot_values_pre):
                        for x in range(len(slot_values_gold)):
                            slot_pre.append(slot_values_pre[x])
                    else:
                        for x in slot_values_pre:
                            slot_pre.append(x)
                        for x in range(len(slot_values_gold) - len(slot_values_pre)):
                            slot_pre.append('O')
            else:
                intent_pred.append(len(intent_set))
                for x in range(len(slot_values_gold)):
                    slot_pre.append('O')
        else:
            intent_pred.append(len(intent_set))
            for x in range(len(slot_values_gold)):
                slot_pre.append('O')

        # print("groundtruth:", groundtruth)
        assert len(slot_pre) == len(slot_gold)

        # print("intent_pred:", intent_pred)
        # print("intent_gold:", intent_gold)


    intent_acc = accuracy_score(intent_gold, intent_pred) * 100

    # calcuate f1 score
    # slot_f1 = f1_score(y2_list, slot_pred, average="macro")
    lines = []
    for pred_index, gold_index in zip(slot_pre, slot_gold):
        # pred_slot = index2slot[pred_index]
        # gold_slot = index2slot[gold_index]
        lines.append("w" + " " + gold_index + " " + pred_index)
    results = conll2002_measure(lines)
    slot_f1 = results["fb1"]
    logger.info("finishing evaluating. request_acc {}, joint_f1 {}, avg_acc {}".format(
        str(intent_acc), str(slot_f1), (intent_acc + slot_f1) / 2))

    return (intent_acc + slot_f1) / 2


def main():
    args = setup_train_args()
    global logger
    logger = create_logger(args)
    test_language = 'en'

    predictions = open(args.result_file_name, "r", encoding='utf-8').read().split('\n\n')
    prediction_dialogues = []
    groundtruth_dialogues = []
    outputs = []
    dialogue_groundtruths = []
    for datas in predictions:
        prediction_dialogues += datas.split('\n')
    prediction_dialogues = prediction_dialogues[0:-1]
    groundtruth = open(args.test_file_name, "r", encoding='utf-8').read().split('\n\n')
    for datas in groundtruth:
        groundtruth_dialogues += datas.split('\n')
    groundtruth_dialogues = groundtruth_dialogues[0:-1]
    assert len(predictions) == len(groundtruth)
    for index in range(len(prediction_dialogues)):
        generation = prediction_dialogues[index].split('<|endofcontext|> ')[1].split(
            '<|endoftext|> ')[0]
        # generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
        dialogue_groundtruth = groundtruth_dialogues[index].split('<|endofcontext|> ')[1].split('<|endoftext|> ')[0]
        outputs.append(generation)
        dialogue_groundtruths.append(dialogue_groundtruth)

    evaluate_result(outputs, dialogue_groundtruths)


if __name__ == '__main__':
    #datas = open('D:/QQDOC/beliefinput2delex_it_test.txt', "r")
    main()


