import logging

import argparse
import numpy as np


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_language', default='en', type=str, required=False, help='target language')
    parser.add_argument('--test_file_name', default='data/beliefinput2delex_en_test.txt', type=str, required=False, help='原始测试语料')
    parser.add_argument('--result_file_name', default='output/pretrained_mt5_beliefinput2delex_en_test.txt', type=str, required=False, help='保存生成结果的路径')
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

def TextToDict(text,args):
    state = {}
    if args.target_language == 'en':
        state = {'food':'none', 'price range':'none', 'request':[], 'area':'none'}
    if args.target_language == 'de':
        state = {'essen': 'none', 'preisklasse': 'none', 'request': [], 'gegend': 'none'}
    if args.target_language == 'it':
        state = {'cibo': 'none', 'prezzo': 'none', 'request': [], 'area': 'none'}
    temp = text.strip()
    for slots in temp.split(','):
        if 'request' in slots:
            state['request'].append(slots.split('slot')[1].strip())
        else:
            for slot in ['food', 'price range', 'area', 'essen', 'preisklasse', 'gegend', 'cibo', 'prezzo']:
                if slot in slots:
                    state[''+slot] = slots.split(slot)[1].strip()
    state['request'] = list(set(state['request']))
    return state


def evaluate_result(evaluated_dialogues, language):
    request = []
    joint = []
    en_slot = ['food', 'price range', 'area']
    de_slot = ['essen', 'preisklasse', 'gegend']
    it_slot = ['cibo', 'prezzo', 'area']

    language_slot = ''
    if language == 'en':
        language_slot = en_slot
    elif language == 'de':
        language_slot = de_slot
    else:
        language_slot = it_slot

    for dialogue in evaluated_dialogues:
        dia = dialogue['dialogue']
        for turn in dia:
            True_state = turn[1]['True State']
            Prediction = turn[2]['Prediction']
            request_gold = set(True_state['request'])
            request_pre = set(Prediction['request'])
            joint_gold = []
            joint_pre = []

            for slot in language_slot:
                joint_gold.append(True_state[slot])
                joint_pre.append(Prediction[slot])
            request.append(request_gold == request_pre)
            joint.append(joint_gold == joint_pre)
    logger.info("final test request_acc:" + str(np.mean(request)))
    logger.info("final test joint_acc:" + str(np.mean(joint)))


def main():
    args = setup_train_args()
    global logger
    logger = create_logger(args)
    test_language = 'en'

    predictions = open(args.result_file_name, "r", encoding='utf-8').read().split('\n\n')
    prediction_dialogues = []
    groundtruth_dialogues = []
    for datas in predictions:
        prediction_dialogues += datas.split('\n')
    prediction_dialogues = prediction_dialogues[0:-1]
    groundtruth = open(args.test_file_name, "r", encoding='utf-8').read().split('\n\n')
    for datas in groundtruth:
        groundtruth_dialogues += datas.split('\n')
    groundtruth_dialogues = groundtruth_dialogues[0:-1]
    assert len(predictions) == len(groundtruth)
    for index in range(len(prediction_dialogues)):
        generation = prediction_dialogues[index].split('<|belief|>')[1].split(
            '<|endofbelief|>')[0]
        # generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
        dialogue_groundtruth = groundtruth_dialogues[index].split('<|belief|>')[1].split('<|endofbelief|>')[0]
        dialogue_input = groundtruth_dialogues[index].split('<|context|>')[1].split('<|endofcontext|>')[0]
        temp_list = []
        true_state = {}
        prediction = {}
        true_state['True State'] = TextToDict(dialogue_groundtruth, args)
        prediction['Prediction'] = TextToDict(generation, args)
        temp_list.append(dialogue_input)
        temp_list.append(true_state)
        temp_list.append(prediction)

        dialogue_dict['dialogue'].append(temp_list)
    dialogue_all.append(dialogue_dict)

    evaluate_result(dialogue_all, args.target_language)


if __name__ == '__main__':
    #datas = open('D:/QQDOC/beliefinput2delex_it_test.txt', "r")
    main()


