from torch.backends import cudnn
from modeling_mt5_mld import MT5ForConditionalGeneration
from transformers import MT5Tokenizer
import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from os.path import join
from dataset_nlg import MyDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import shutil
import re
import time
import pickle
import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge

from torch.nn import CrossEntropyLoss
import pdb
PAD = '[PAD]'
pad_id = 0
logger = None
CODE_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# local_rank = int(os.environ["LOCAL_RANK"])
#import torch
# torch.cuda.current_device()
# torch.cuda._initialized = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def ontology_mapping(match_language, string):
    if match_language == 'de':
        f = open('../data/dst_vocab/en2de_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'es ist egal'
    else:
        f = open('../data/dst_vocab/en2it_onto_for_mix.dict', 'rb')
        de_mapping = pickle.load(f)
        de_mapping['dontcare'] = 'non importa'
        #de_mapping['price'] = 'prezzo'

    punctuation = [',', '?', '.']
    for punc in punctuation:
        string = string.replace(punc, ' ' + punc)

    for key, value in de_mapping.items():
        if len(key.split()) > 1:
            if key == "price range":  ## could be price ranges in the utterance
                string = string.replace("price ranges", value)

            string = string.replace(key, value)

        else:
            splits_user = string.split()
            for i, word in enumerate(splits_user):
                if word == key: splits_user[i] = value
            string = " ".join(splits_user)

    return string

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_language', default='en', type=str, required=False, help='source hidden language')
    parser.add_argument('--target_language', default='en', type=str, required=False, help='target language')

    parser.add_argument('--epochs', default=60, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--gradient_accumulation', default=4, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--seed', type=int, default=12345, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--continue_epoch', default=0, type=int, required=False, help='set start epoch')

    parser.add_argument('--root_dir_data', default='%s/data' % CODE_ROOT_DIR, type=str, required=False,
                        help='Root directory of all data，e.g., dataset, vocab, pretrained models')
    parser.add_argument('--root_dir_output', default='%s/output' % CODE_ROOT_DIR, type=str, required=False,
                        help='Root directory of all output，e.g., model, result')
    parser.add_argument('--train_file_name', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--val_file_name', default='data/val.txt', type=str, required=False, help='原始验证语料')
    parser.add_argument('--test_file_name', default='data/test.txt', type=str, required=False, help='原始测试语料')
    parser.add_argument('--save_file_name', default='output/pretrained_mt5.txt', type=str, required=False, help='保存生成结果的路径')
    parser.add_argument('--ontology_path', default='ontology/ontology_dstc2_en.json', type=str, required=False,
                        help='存放slot定义的路径')
    parser.add_argument('--exp_name', default='', type=str, required=False, help='experiment name')

    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='训练日志存放位置')

    parser.add_argument('--eval_all_checkpoints', action='store_true', help='在所有模型上评价')
    parser.add_argument('--prefix', action='store_true', required=False, help='train with prefix')
    parser.add_argument('--change_parameter', action='store_true', required=False, help='change parameter')
    parser.add_argument("--local_rank", type=int, default=-1, help='distributed')
    parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--mix_train', action='store_true', help='train using mix language')
    parser.add_argument('--mode', default='train', type=str, required=False, help='train using mix language')
    parser.add_argument('--evaluate_type', default='avg_acc', type=str, required=False, help='selct a evaluate method: loss or avg_acc')
    parser.add_argument('--cross', action='store_true', help='cross datas')
    return parser.parse_args()


def set_random_seed(args):
    """
    设置训练的随机种子
    """
    seed = args.seed
    if seed is None:
        ms = time.time() * 1000
        seed = int(ms // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True

def compare_request_lists(list_a, list_b):

    if len(list_a) != len(list_b):
        return False

    list_a.sort()
    list_b.sort()

    for idx in range(0, len(list_a)):
        if list_a[idx] != list_b[idx]:
            return False

    return True

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

def create_model(args):
    """

    :param args:
    :param vocab_size:字典大小
    :return:
    """
    if args.pretrained_model:
        model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)
        '''if args.change_parameter:
            mapping = torch.zeros(model.config.hidden_size,model.config.hidden_size)
            torch.nn.init.xavier_uniform_(mapping)
            model_dict = model.state_dict()
            model_dict['mapping.weight'] = mapping
            model = MT5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=None, config=model.config, state_dict=model_dict)'''
    elif args.continue_epoch > 0:
        previous_model = '%s/model/%s_model/model_epoch%s/' % (args.root_dir_output, args.exp_name, args.continue_epoch - 1)
        model = MT5ForConditionalGeneration.from_pretrained(previous_model)
    else:
        model = MT5ForConditionalGeneration.from_pretrained("/data/yanguojun-slurm/GPT2-chitchat/model/mt5/")
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model
def collate_fn(batch):
    """train
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    btc_size = len(batch)

    inputs = []
    labels = []
    for btc_idx in range(btc_size):
        #inputs.append('<dst> <en>: '+batch[btc_idx].split("<|endofcontext|>")[0].split("<|context|>")[1])
        inputs.append(batch[btc_idx][0])
        labels.append(batch[btc_idx][1])
    return [inputs, labels]

def train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer):
    candidates = '<|candidates|> request address,request area,request food,request phone,request price range,' \
                 'request postcode,request name,food afghan,food african,food afternoon tea,food asian oriental,' \
                 'food australasian,food australian,food austrian,food barbeque,food basque,food belgian,food bistro,' \
                 'food brazilian,food british,food canapes,food cantonese,food caribbean,food catalan,food chinese,' \
                 'food christmas,food corsica,food creative,food crossover,food cuban,food danish,food eastern european,' \
                 'food english,food eritrean,food european,food french,food fusion,food gastropub,food german,food greek,' \
                 'food halal,food hungarian,food indian,food indonesian,food international,food irish,food italian,food jamaican,' \
                 'food japanese,food korean,food kosher,food latin american,food lebanese,food light bites,food malaysian,food mediterranean,' \
                 'food mexican,food middle eastern,food modern american,food modern eclectic,food modern european,food modern global,' \
                 'food molecular gastronomy,food moroccan,food new zealand,food north african,food north american,food north indian,' \
                 'food northern european,food panasian,food persian,food polish,food polynesian,food portuguese,food romanian,food russian,' \
                 'food scandinavian,food scottish,food seafood,food singaporean,food south african,food south indian,food spanish,food sri lankan,' \
                 'food steakhouse,food swedish,food swiss,food thai,food the americas,food traditional,food turkish,food tuscan,food unusual,food vegetarian,' \
                 'food venetian,food vietnamese,food welsh,food world,price range cheap,price range moderate,price range expensive,area centre,area north,' \
                 'area west,area south,area east <|endofcandidates|>'

    print(len(train_list))
    train_datasets = []
    train_dataset = MyDataset(train_list)
    #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
    model.train()
    # 计算所有epoch进行参数优化的总步数total_steps

    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=total_steps)

    logger.info('starting training')
    # 用于统计每次梯度累计的loss

    # 记录 out of memory的次数
    oom_time = 0
    #temp1 = model.state_dict()['module.lm_head.weight'][0, 10]
    #temp2 = model.state_dict()['module.lm_head.weight'][250101, 10]
    # 开始训练
    for epoch in range(args.continue_epoch,args.epochs):
        running_loss = 0
        # 统计一共训练了多少个step
        overall_step = 1
        if torch.cuda.is_available():
                sampler = DistributedSampler(train_dataset)
                sampler.set_epoch(epoch)
                train_dataloaders = []

                train_dataloader =DataLoader(train_dataset, batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      collate_fn=collate_fn, sampler=sampler)

        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers,
                                          collate_fn=collate_fn)
        epoch_start_time = datetime.now()
        for batch_idx, input_ids in enumerate(train_dataloader):
            print("inputs-dataset:", input_ids)
            inputs = tokenizer(input_ids[0], return_tensors="pt",padding=True).to(device)
            #candidate_inputs = tokenizer([candidates] * inputs['input_ids'].size(0), return_tensors="pt",
            #                             padding=True).to(device)
            with tokenizer.as_target_tokenizer():
                if input_ids[1] is not None:
                    labels = tokenizer(input_ids[1], return_tensors="pt", padding=True).to(device)
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                outputs = model(**inputs, labels=labels["input_ids"], parameter_id=0)
                #candidate_outputs = model(**candidate_inputs, labels=labels['input_ids'])
                #print("inputs:",input_ids[0])
                #print("labels:", input_ids[1])

                #logits = outputs.logits
                #logits = logits + torch.mul(candidate_outputs.logits, outputs.logits)
                #pdb.set_trace()

                #loss_fct = CrossEntropyLoss(ignore_index=-100)
                #loss = loss_fct(logits.view(-1, logits.size(-1)), labels['input_ids'].view(-1))

                loss = outputs.loss
                #print("loss:", loss)
                if multi_gpu:
                    loss = loss.mean()
                if args.gradient_accumulation > 1:
                    loss = loss.mean() / args.gradient_accumulation
                #loss.backward(loss.clone().detach())
                loss.backward()
                # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # 进行一定step的梯度累计之后，更新参数
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.mean().item()
                    # 更新参数
                    optimizer.step()
                    # 清空梯度信息
                    optimizer.zero_grad()
                    # 进行warm up
                    scheduler.step()
                    overall_step += 1
                    # 更新日志与tnesorboardX信息
                    if (overall_step + 1) % args.log_step == 0 and args.local_rank == 0:
                        logger.info(
                            "batch {} of epoch {}, loss {}".format(
                                batch_idx + 1, epoch + 1, loss.mean()))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception
        if args.local_rank == 0:
            logger.info('saving model for epoch {}'.format(epoch + 1))
            model_path = join(args.dialogue_model_output_path, 'model_epoch{}'.format(epoch + 1))
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)
            tb_writer.add_scalar('train_loss', running_loss / overall_step, epoch+1)
            logger.info('epoch {} finished'.format(epoch + 1))
            logger.info("finishing evaluating. loss {}".format(
                running_loss / overall_step))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    logger.info('training finished')


def evaluate_loss(model, device, val_list, multi_gpu, args, tokenizer, tb_writer, overstep):
    candidates = '<|candidates|> request address,request area,request food,request phone,request price range,' \
                 'request postcode,request name,food afghan,food african,food afternoon tea,food asian oriental,' \
                 'food australasian,food australian,food austrian,food barbeque,food basque,food belgian,food bistro,' \
                 'food brazilian,food british,food canapes,food cantonese,food caribbean,food catalan,food chinese,' \
                 'food christmas,food corsica,food creative,food crossover,food cuban,food danish,food eastern european,' \
                 'food english,food eritrean,food european,food french,food fusion,food gastropub,food german,food greek,' \
                 'food halal,food hungarian,food indian,food indonesian,food international,food irish,food italian,food jamaican,' \
                 'food japanese,food korean,food kosher,food latin american,food lebanese,food light bites,food malaysian,food mediterranean,' \
                 'food mexican,food middle eastern,food modern american,food modern eclectic,food modern european,food modern global,' \
                 'food molecular gastronomy,food moroccan,food new zealand,food north african,food north american,food north indian,' \
                 'food northern european,food panasian,food persian,food polish,food polynesian,food portuguese,food romanian,food russian,' \
                 'food scandinavian,food scottish,food seafood,food singaporean,food south african,food south indian,food spanish,food sri lankan,' \
                 'food steakhouse,food swedish,food swiss,food thai,food the americas,food traditional,food turkish,food tuscan,food unusual,food vegetarian,' \
                 'food venetian,food vietnamese,food welsh,food world,price range cheap,price range moderate,price range expensive,area centre,area north,' \
                 'area west,area south,area east <|endofcandidates|>'

    logger.info("start evaluating model")
    model.eval()
    # 记录tensorboardX
    loss_all = 0
    step_all = 0
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    val_dataset = MyDataset(val_list)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
    with torch.no_grad():

        for batch_idx, input_ids in enumerate(val_dataloader):
            #special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            #input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)
            inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True)
            #candidate_inputs = tokenizer([candidates] * inputs['input_ids'].size(0), return_tensors="pt",
            #                             padding=True).to(device)
            inputs = inputs.to(device)

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(input_ids[1], return_tensors="pt", padding=True)
                labels = labels.to(device)

                outputs = model(**inputs, labels=labels["input_ids"])
                #candidate_outputs = model(**candidate_inputs, labels=labels['input_ids'])

                #logits = outputs.logits
                #logits = logits + torch.mul(candidate_outputs.logits, outputs.logits)


                #loss_fct = CrossEntropyLoss(ignore_index=-100)
                #loss = loss_fct(logits.view(-1, logits.size(-1)), labels['input_ids'].view(-1))
                loss = outputs.loss
                loss_all += loss.mean()
                overstep[0] += 1
                step_all += 1

                if multi_gpu:
                    loss = loss.mean()

                if args.gradient_accumulation > 1:
                    loss = loss.mean() / args.gradient_accumulation

                if (batch_idx % args.log_step) == 0:
                    logger.info(
                        "evaluate batch {} ,loss {}".format(
                            batch_idx, loss.mean()))
                if args.local_rank == 0:
                    tb_writer.add_scalar('valid_loss', loss.mean().item(), overstep[0])
        batch_num = len(val_dataloader)
        logger.info("finishing evaluating. loss {}".format(
            loss_all / step_all))

    return loss_all/ step_all

def evaluate_avg_acc(model, device,val_list, args, tokenizer, tb_writer, overstep):
    logger.info("start evaluating model")
    model.eval()
    # 记录tensorboardX
    loss_all = 0
    step_all = 0
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    val_dataset = MyDataset(val_list)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*10, shuffle=True, num_workers=args.num_workers,
                                collate_fn=collate_fn)
    with torch.no_grad():

        request = []
        joint = []
        for batch_idx, input_ids in enumerate(val_dataloader):
            # special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            # input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)
            inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True)
            # candidate_inputs = tokenizer([candidates] * inputs['input_ids'].size(0), return_tensors="pt",
            #                             padding=True).to(device)
            inputs = inputs.to(device)
            dialogue_groundtruth = input_ids[1]
            outputs = model.generate(inputs["input_ids"], max_length=200, parameter_id=0)

            for index in range(len(outputs)):
                groundtruth = dialogue_groundtruth[index].split('<|belief|>')[1].split(
                    '<|endofbelief|>')[0]
                #print("groundtrhtu:", groundtruth)
                try:
                    generation = tokenizer.decode(outputs[index])
                    if '<|belief|>' in generation and '<|endofbelief|>' in generation:
                        generation = \
                        generation.split('<|belief|>')[1].split(
                            '<|endofbelief|>')[0]
                        if args.cross:
                            generation = ontology_mapping(args.target_language, generation)
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

                request.append(request_gold == request_pred)
                joint.append(joint_gold == joint_pred)

            overstep[0] += 1
            step_all += 1



            if (batch_idx % args.log_step) == 0:
                logger.info(
                    "evaluate batch {} ,request_acc {}, joint_acc {}, avg_acc {}".format(
                        batch_idx, np.mean(request), np.mean(joint), (np.mean(request)+np.mean(joint))/2))
            if args.local_rank == 0:
                tb_writer.add_scalar('avg_acc', (np.mean(request)+np.mean(joint))/2, overstep[0])
        logger.info("finishing evaluating. request {}, joint {}, avg_acc {}".format(np.mean(request), np.mean(joint),
                     (np.mean(request) + np.mean(joint)) / 2))

    return (np.mean(request)+np.mean(joint))/2
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
def generate(model,tokenizer,test_list,args,device):
    logger.info('starting generating')
    save_path = open(args.save_file_name, 'w', encoding='utf-8')

    joint_acc = 0
    model.eval()
    count = 0
    dialogue_all = []

    for dialogue in test_list:

        dialogue_dict = {}
        count += 1
        # process dialogue
        dialogue_inputs = []
        dialogue_groundtruth = []
        dialogue_groundtruth.append(dialogue[1])
        dialogue_inputs.append(dialogue[0])
        print("dialogue_inputs:", dialogue_inputs)
        # model generate

        inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(inputs["input_ids"], max_length=200,parameter_id=0)

        #print(outputs)

        # tokenizer decode and
        dialogue_input_new = ''
        for index in range(len(outputs)):
            print(tokenizer.decode(outputs[index]))
            generation = \
            re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))

            temp_list = []
            temp_list.append(generation)
            temp_list.append(dialogue_groundtruth[index])
            dialogue_all.append(temp_list)

    json.dump(dialogue_all, save_path, indent=1)
    save_path.close()

def evaluate_result(evaluated_dialogues):
    eval_preds = json.load(open(evaluated_dialogues, 'r', encoding='utf-8'))
    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for sentence in eval_preds:
        pred, label = sentence
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
            result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        else:
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))

        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))
    print({k: float(np.mean(v)) for k, v in score_dict.items()})

def init(args):
        args.dialogue_model_output_path = args.root_dir_output+'/model/' + args.dialogue_model_output_path
        args.save_file_name = args.root_dir_output+'/result/' + args.save_file_name
        args.writer_dir = CODE_ROOT_DIR+'/tensorboard_summary/'+args.writer_dir
        # 创建对话模型的输出目录
        if not os.path.exists(args.dialogue_model_output_path):
            os.mkdir(args.dialogue_model_output_path)
        # 记录tensorboardX
        if os.path.exists(args.writer_dir):
            shutil.rmtree(args.writer_dir)
        os.mkdir(args.writer_dir)
        tb_writer = SummaryWriter(log_dir=args.writer_dir)
        # 是否使用多块GPU进行并行运算
        multi_gpu = False
        if torch.cuda.device_count() > 1:
            multi_gpu = True
        if len(args.train_file_name.split(','))>1:
            temp = ''
            for x in args.train_file_name.split(','):
                temp += args.root_dir_data+'/mulwoz_process/' + x +','
            args.train_file_name = temp[:-1]
        else:
            args.train_file_name = args.root_dir_data+'/mulwoz_process/' + args.train_file_name
        args.val_file_name = args.root_dir_data+'/mulwoz_process/' + args.val_file_name
        args.test_file_name = args.root_dir_data+'/mulwoz_process/' + args.test_file_name
        args.ontology_path = args.root_dir_data+'/ontology/ontology_dstc2_%s.json' % args.target_language

        return tb_writer, multi_gpu


def main():
    args = setup_train_args()

    tb_writer = ''
    multi_gpu = ''
    # 日志同时输出到文件和console
    global logger
    logger = create_logger(args)

    # Setup CUDA, GPU & distributed training
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'

    print(args.cuda)
    #args.local_rank = local_rank
    device = torch.device("cuda", args.local_rank)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    logger.info('using device:{}'.format(device))
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if args.seed:
        set_random_seed(args)

    # tokenizer的字典大小
    model = create_model(args)
    tokenizer = MT5Tokenizer.from_pretrained("/data/yanguojun-slurm/GPT2-chitchat/model/mt5/")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|system|>','<en>', '<dst>','<de>','<it>'
                                                                  ,'<|number|>','<|belief|>' ,'<|endofbelief|>']})


    vocab_size = len(tokenizer)
    global pad_id
    pad_id = PAD+' '
    model.to(device)

    if args.local_rank == 0:

        # 记录模型参数数量
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        logger.info('number of model parameters: {}'.format(num_parameters))
        # 对原始数据进行预处理,将原始语料转换成对应的token_id
        tb_writer, multi_gpu = init(args)
        print("args:", args)

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # 加载数据
    logger.info("loading traing data")
    # consider train_path list
    train_list = []
    if len(args.train_file_name.split(',')) == 1:
         train_temp = json.load(open(args.train_file_name, "r", encoding='utf-8'))
         contexts = train_temp['contexts']
         responses = train_temp['responses']
         for i in range(len(contexts)):
            train_list.append([contexts[i], responses[i]])
    else:
         train_list = []
         for x in args.train_file_name.split(','):
             temp_language = x.split('_')[3]
             print("train_temp_language:", temp_language)
             train_temp = json.load(open(x, "r", encoding='utf-8'))
             contexts = train_temp['contexts']
             responses = train_temp['responses']
             for i in range(len(contexts)):
                 train_list.append([contexts[i], responses[i]])
    logger.info("loading val data")
    val_temp = json.load(open(args.val_file_name, "r", encoding='utf-8'))
    val_list = []
    temp_language = args.val_file_name.split('_')[3]
    print("dev_temp_language:", temp_language)
    if args.prefix:
        try:
            contexts = val_temp['contexts']
            responses = val_temp['responses']
            for i in range(len(contexts)):
                val_list.append([contexts[i], responses[i]])
        except IndexError:
            print("load val dataset error")
    else:
        contexts = val_temp['contexts']
        responses = val_temp['responses']
        for i in range(len(contexts)):
            val_list.append([contexts[i], responses[i]])
    logger.info("loading testing data")

    test_temp = json.load(open(args.test_file_name, "r", encoding='utf-8'))
    temp_language = args.test_file_name.split('_')[3]
    print("test_language:", temp_language)
    test_list = []
    if args.prefix:
        try:
            contexts = test_temp['contexts']
            responses = test_temp['responses']
            for i in range(len(contexts)):
                test_list.append([contexts[i], responses[i]])
        except IndexError:
            print("load val dataset error")
    else:
        contexts = test_temp['contexts']
        responses = test_temp['responses']
        for i in range(len(contexts)):
            test_list.append([contexts[i], responses[i]])
    # 开始训练

    if args.mode == 'train':
        train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer)
        # 模型验证
        if args.local_rank == 0:
            best_model = ''
            if args.eval_all_checkpoints:
                checkpoints = [args.dialogue_model_output_path + c for c in
                               sorted(os.listdir(args.dialogue_model_output_path))[1:-1]]
                logger.info("Evaluate the following checkpoints: {}".format(checkpoints))
                overstep = [0]

                score = 0
                min_loss = 10000
                for x in range(1, args.epochs + 1):
                    checkpoint = args.dialogue_model_output_path + 'model_epoch' + str(x)
                    model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
                    logger.info("Evaluate the checkpoint: {}".format(checkpoint))
                    model.resize_token_embeddings(vocab_size)
                    model.to(device)
                    if args.evaluate_type == 'avg_acc':
                        result = evaluate_avg_acc(model, device, val_list, args, tokenizer, tb_writer, overstep)
                        if result > score:
                            score = result
                            best_model = checkpoint
                    else:
                        result = evaluate_loss(model, device, val_list, multi_gpu, args, tokenizer, tb_writer, overstep)
                        if result < min_loss:
                            min_loss = result
                            best_model = checkpoint
                logger.info("the best model is " + best_model)
            tb_writer.close()
            if args.mix_train:
                test_list = open(args.train_file_name, "r", encoding='utf-8').read().split('\n\n')[0:-1]
                min_model = '../output/model/MT5_mot_belief_ende_en_generate_test_model/model_epoch55/'
                model = MT5ForConditionalGeneration.from_pretrained(min_model)
                # model_dict = model.state_dict()
                model.to(device)
                generate(model, tokenizer, test_list, args, device)
                evaluate_result(args.save_file_name)
            else:
                #MT5_mot_belieturn_en_model/model_epoch46/
                #min_model = '../output/model/MT5_mot_beliefturncross_en_model/model_epoch14/'
                model = MT5ForConditionalGeneration.from_pretrained(best_model)
                #model_dict = model.state_dict()
                model.to(device)
                overstep = [0]
                generate(model, tokenizer, test_list, args, device)
                evaluate_result(args.save_file_name)
    elif args.mode == 'eva':
        best_model = ''
        if args.eval_all_checkpoints:
            checkpoints = [args.dialogue_model_output_path + c for c in
                           sorted(os.listdir(args.dialogue_model_output_path))[1:-1]]
            logger.info("Evaluate the following checkpoints: {}".format(checkpoints))
            overstep = [0]

            score = 0
            min_loss = 10000
            for x in range(1, args.epochs + 1):
                checkpoint = args.dialogue_model_output_path + 'model_epoch' + str(x)
                model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
                logger.info("Evaluate the checkpoint: {}".format(checkpoint))
                model.resize_token_embeddings(vocab_size)
                model.to(device)
                if args.evaluate_type == 'avg_acc':
                    result = evaluate_avg_acc(model, device, val_list, args, tokenizer, tb_writer, overstep)
                    if result > score:
                        score = result
                        best_model = checkpoint
                else:
                    result = evaluate_loss(model, device, val_list, multi_gpu, args, tokenizer, tb_writer, overstep)
                    if result < min_loss:
                        min_loss = result
                        best_model = checkpoint
            logger.info("the best model is " + best_model)
        tb_writer.close()
        model = MT5ForConditionalGeneration.from_pretrained(best_model)
        # model_dict = model.state_dict()
        model.to(device)
        generate(model, tokenizer, test_list, args, device)
        evaluate_result(args.save_file_name)
    else:
        if args.local_rank == 0:
            #min_model = '../output/model/MT5_mot_belieturn_en_model/model_epoch46/'
            model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)
            # model_dict = model.state_dict()
            model.to(device)
            generate(model, tokenizer, test_list, args, device)
            evaluate_result(args.save_file_name)
if __name__ == '__main__':
    main()
