from torch.backends import cudnn
from transformers import MT5ForConditionalGeneration
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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import shutil
import re
import time
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score
from conll2002_metrics import *
import pdb
PAD = '[PAD]'
pad_id = 0
logger = None
CURRENT_LANG = ''
CODE_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class MyDataset(Dataset):
    """

    """

    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.langs = ['<en>', '<es>', '<th>']

    def __getitem__(self, index):
        inputs = self.data_list[index].strip()
        if '<|trans|>' in inputs:
            inputs_com = inputs.split("<|trans|>")[0].split("<|context|>")[1] + '<|endofcontext|>'
            labels = inputs.split("<|endofcontext|>")[1].split("<|slot_values|>")[0]
            intent_labels = inputs.split("<|intent|>")[1].split("<|slot_value|>")[0]
            slot_labels = inputs.split("<|slot_value|>")[1].split("<|endof_slot_value|>")[0].strip()
            #slots = inputs.split("<|slot_values|>")[1].split("<|endoftext|>")[0].strip()
            for lang in self.langs:
                input_src = inputs.split("<|trans|>")[0].replace(lang, '')

            input_src = input_src.split("<|trans|>")[0].split("<|context|>")[1]
            input_tgt = '<nlu> ' + inputs.split("<|mlt|>")[0].split("<|trans|>")[1]
            #input_mlt = inputs.split("<|mlt|>")[1].split("<|endofcontext|>")[0]
            input_mlt = inputs_com.split(input_src.split("<nlu>")[1])[0] + inputs.split("<|mlt|>")[1].split \
                ("<|endofcontext|>")[0] + '<|endofcontext|>'
            #print(input_mlt)
            return [inputs_com, labels, input_src, input_tgt, intent_labels, input_mlt, slot_labels]
        else:
            inputs_com = inputs.split('<|endofcontext|>')[0].split("<|context|>")[1] + ' <|endofcontext|>'
            labels = inputs.split("<|endofcontext|>")[1].split("<|endoftext|>")[0]
            return [inputs_com, labels]

    def __len__(self):
        return len(self.data_list)

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
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
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
    parser.add_argument('--local_test', action='store_true', help='local test')
    parser.add_argument('--cross', action='store_true', help='cross_lingual')
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
        logger.info("previous_model:{}".format(previous_model))
        #print("previous_model:", previous_model)
        model = MT5ForConditionalGeneration.from_pretrained(previous_model)
    else:
        if args.local_test:
            model = MT5ForConditionalGeneration.from_pretrained("../../mt5/")
        else:
            model = MT5ForConditionalGeneration.from_pretrained("../../GPT2-chitchat/model/mt5/")
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
    input_src = []
    input_tgt = []
    input_mlt = []
    intent_labels = []
    slot_lables = []
    slots = []
    # if len(batch[0]) == 4:
    #     for btc_idx in range(btc_size):
    #         inputs.append(batch[btc_idx].split("<|trans|>")[0].split("<|context|>")[1] + '<|endofcontext|>')
    #         labels.append(batch[btc_idx].split("<|endofcontext|>")[1].split("<|endoftext|>")[0])
    #         input_src.append('<nlu> '+batch[btc_idx].split("<|trans|>")[0].split(CURRENT_LANG)[2])
    #         input_tgt.append('<nlu> ' + batch[btc_idx].split("<|endofcontext|>")[0].split("<|trans|>")[1])
    #     return [inputs, labels, input_src, input_tgt]
    # else:
    #     for btc_idx in range(btc_size):
    #         inputs.append(batch[btc_idx].split('<|endofcontext|>')[0].split("<|context|>")[1] + ' <|endofcontext|>')
    #         labels.append(batch[btc_idx].split("<|endofcontext|>")[1].split("<|endoftext|>")[0])
    #     return [inputs, labels]
    if len(batch[0]) == 2:
        for btc_idx in range(btc_size):
            inputs.append(batch[btc_idx][0])
            labels.append(batch[btc_idx][1])
        return [inputs, labels]
    else:
        for btc_idx in range(btc_size):
            inputs.append(batch[btc_idx][0])
            labels.append(batch[btc_idx][1])
            input_src.append(batch[btc_idx][2])
            input_tgt.append(batch[btc_idx][3])
            intent_labels.append(batch[btc_idx][4])
            input_mlt.append(batch[btc_idx][5])
            slot_lables.append(batch[btc_idx][6])
            #slots.append(batch[btc_idx][7])
        return [inputs, labels, input_src, input_tgt, intent_labels, input_mlt, slot_lables, slots]
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
    encoder = model.module.get_encoder()
    print(len(train_list))
    train_datasets = []
    train_dataset = MyDataset(train_list, tokenizer)
    #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
    model.train()
    # 计算所有epoch进行参数优化的总步数total_steps

    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.08 * total_steps, num_training_steps=total_steps)

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
            #print(input_ids)

            inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True).to(device)
            input_src = tokenizer(input_ids[2], return_tensors="pt", padding=True).to(device)
            input_tgt = tokenizer(input_ids[3], return_tensors="pt", padding=True).to(device)
            intent_labels = tokenizer(input_ids[4], return_tensors="pt", padding=True).to(device)
            input_mlt = tokenizer(input_ids[5], return_tensors="pt", padding=True).to(device)
            slot_labels = tokenizer(input_ids[6], return_tensors="pt", padding=True).to(device)
            #slots = tokenizer(input_ids[7], return_tensors="pt", padding=True).to(device)
            #candidate_inputs = tokenizer([candidates] * inputs['input_ids'].size(0), return_tensors="pt",
            #                             padding=True).to(device)
            with tokenizer.as_target_tokenizer():
                if input_ids[1] is not None:
                    labels = tokenizer(input_ids[1], return_tensors="pt", padding=True).to(device)
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                # origin input and label
                if args.local_test:
                    outputs = model(**inputs, labels=labels["input_ids"],  return_dict=True)

                    # alignment loss
                    inputs_src_last_encoder_hidden_state = model(**input_src, labels=labels["input_ids"], return_dict=True)\
                        .encoder_last_hidden_state

                    inputs_tgt_last_encoder_hidden_state = model(**input_tgt, labels=labels["input_ids"],  return_dict=True) \
                        .encoder_last_hidden_state
                else:
                    # nlu_loss
                    #outputs = model(**inputs, labels=labels["input_ids"], parameter_id=0, return_dict=True)

                    # alignment loss

                    # origin_input
                    # inputs_src_last_encoder_hidden_state = model(**input_src, labels=labels["input_ids"],
                    #                                              parameter_id=0, return_dict=True) \
                    #     .encoder_last_hidden_state
                    # inputs_src_last_encoder_hidden_state = encoder(**input_src, return_dict=True) \
                    #                                                  .last_hidden_state
                    # inputs_tgt_last_encoder_hidden_state = encoder(**input_tgt, return_dict=True) \
                    #     .last_hidden_state
                    # translated_input
                    # inputs_tgt_last_encoder_hidden_state = model(**input_tgt, labels=labels["input_ids"],
                    #                                              parameter_id=0, return_dict=True) \
                    #     .encoder_last_hidden_state
                    tgt_loss = model(**input_tgt, labels=intent_labels["input_ids"],
                                                                 parameter_id=0, return_dict=True) \
                        .loss
                    mlt_loss = model(**input_mlt, labels=labels["input_ids"],
                                                                 parameter_id=0, return_dict=True) \
                        .loss
                    # slot_loss = model(**input_tgt, labels=slots["input_ids"],
                    #                  parameter_id=0, return_dict=True) \
                    #     .loss
                # inputs_src_emb = inputs_src_last_encoder_hidden_state[:, 0, :]
                # inputs_tgt_emb = inputs_tgt_last_encoder_hidden_state[:, 0, :]
                # inputs_src_emb = torch.mean(inputs_src_last_encoder_hidden_state, dim=1)
                # inputs_tgt_emb = torch.mean(inputs_tgt_last_encoder_hidden_state, dim=1)
                #
                # MSE_loss = torch.nn.MSELoss()
                # mse_loss = MSE_loss(inputs_src_emb, inputs_tgt_emb)

                # total loss
                #loss = outputs.loss + 0.05 * abs(30-epoch)*tgt_loss + 0.05 * abs(30-epoch)*mlt_loss
                loss = 0.9*mlt_loss + 0.1 * tgt_loss
                #loss = mse_loss
                #model_dict = model.state_dict()
                #print("loss:", outputs.loss, "intent_loss:", tgt_loss)
                #print("loss:", outputs.loss, "intent_loss:", tgt_loss, "mlt_loss:", mlt_loss)
                #print(model_dict.keys())
                #print("loss:", outputs.loss, "mes_loss:", mse_loss)
                # print("encoder-weight:", model_dict['module.encoder.block.0.layer.0.SelfAttention.q.weight'][0][0])
                # print("decoder-weight:", model_dict['module.decoder.block.0.layer.1.EncDecAttention.k.weight'][0][0])
                if multi_gpu:
                    loss = loss.mean()
                if args.gradient_accumulation > 1:
                    loss = loss.mean() / args.gradient_accumulation
                #loss.backward(loss.clone().detach())
                loss.backward()

                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(name)
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

# def evaluate_avg_acc(model, device,val_list, args, tokenizer, tb_writer, overstep):
#     index2slot = {'O':0, 'B-weather/noun':1, 'I-weather/noun':2, 'B-location':3, 'I-location':4, 'B-datetime':5, 'I-datetime':6,
#                   'B-weather/attribute':7, 'I-weather/attribute':8, 'B-reminder/todo':9, 'I-reminder/todo':10,
#                   'B-alarm/alarm_modifier':11, 'B-reminder/noun':12, 'B-reminder/recurring_period':13,
#                   'I-reminder/recurring_period':14, 'B-reminder/reference':15, 'I-reminder/noun':16,
#                   'B-reminder/reminder_modifier':17, 'I-reminder/reference':18, 'I-reminder/reminder_modifier':19,
#                   'B-weather/temperatureUnit':20, 'I-alarm/alarm_modifier':21, 'B-alarm/recurring_period':22,
#                   'I-alarm/recurring_period':23}
#     logger.info("start evaluating model")
#     model.eval()
#     # 记录tensorboardX
#     loss_all = 0
#     step_all = 0
#     # tb_writer = SummaryWriter(log_dir=args.writer_dir)
#     val_dataset = MyDataset(val_list)
#     val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*10, shuffle=True, num_workers=args.num_workers,
#                                 collate_fn=collate_fn)
#     with torch.no_grad():
#
#         request = []
#         joint = []
#         res_pre = []
#         res_gold = []
#         for batch_idx, input_ids in enumerate(val_dataloader):
#             # special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
#             # input_ids = input_ids[:, 6:].to(device)
#             # input_ids.to(device)
#             inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True)
#             # candidate_inputs = tokenizer([candidates] * inputs['input_ids'].size(0), return_tensors="pt",
#             #                             padding=True).to(device)
#             inputs = inputs.to(device)
#             dialogue_groundtruth = input_ids[1]
#             #parameter_id = parameter_ids[input_ids[0][0].split('<nlu>')[1].split(':')[0].strip()]
#             outputs = model.generate(inputs["input_ids"], max_length=200, parameter_id=0)
#
#
#             for index in range(len(outputs)):
#                 groundtruth = dialogue_groundtruth[index].split('<|intent_slot|>')[1].split(
#                     '<|endof_intent_slot|>')[0]
#                 #print("groundtrhtu:", groundtruth)
#                 try:
#                     generation = tokenizer.decode(outputs[index])
#                     if '<|intent_slot|>' in generation and '<|endof_intent_slot|>' in generation:
#                         generation = \
#                         generation.split('<|intent_slot|>')[1].split(
#                             '<|endof_intent_slot|>')[0]
#                     # generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
#                     else:
#                         generation = 'error'
#                     print('generation:', generation)
#                 except IndexError:
#                     generation = 'error'
#                     #print("error")
#
#                 intent_gold = ''
#                 intent_pred = ''
#                 joint_gold = {}
#                 joint_pred = {}
#                 slot_values_pre = ''
#                 if generation != 'error' and generation != ' ':
#                     if '<|slot_value|>' in generation:
#
#                         intent_pred = generation.split('<|slot_value|>')[0].split('<|intent|>')[1].strip()
#                         slot_values_pre = generation.split('<|slot_value|>')[1].strip().split()
#
#                         for slot_value in slot_values_pre:
#                             if slot_value in index2slot.keys():
#                                 res_pre.append(index2slot[slot_value])
#                             else:
#                                 res_pre.append(len(index2slot))
#
#
#                 intent_gold = groundtruth.split('<|slot_value|>')[0].split('<|intent|>')[1].strip()
#                 slot_values_gold = groundtruth.split('<|slot_value|>')[1].strip().split()
#                 for slot_value in slot_values_gold:
#                     if slot_value in index2slot.keys():
#                         res_gold.append(index2slot[slot_value])
#                     else:
#                         res_gold.append(len(index2slot))
#
#                 print("groundtruth:", groundtruth)
#
#                 request.append(intent_gold == intent_pred)
#                 print("intent_pred:", intent_pred)
#                 print("intent_gold:", intent_gold)
#
#                 if len(slot_values_gold) < len(slot_values_pre):
#                     res_gold += ([len(index2slot)]*(len(slot_values_pre)-len(slot_values_gold)))
#                 elif len(slot_values_gold) > len(slot_values_pre):
#                     res_pre += ([len(index2slot)] * (len(slot_values_gold) - len(slot_values_pre)))
#             overstep[0] += 1
#             step_all += 1
#
#             assert len(res_pre) == len(res_gold)
#             slot_f1 = f1_score(y_true=res_gold, y_pred=res_pre,average='micro')
#             if (batch_idx % args.log_step) == 0:
#                 logger.info(
#                     "evaluate batch {} ,request_acc {}, joint_acc {}, avg_acc {}".format(
#                         batch_idx, np.mean(request), str(slot_f1), (np.mean(request)+slot_f1)/2))
#             if args.local_rank == 0:
#                 tb_writer.add_scalar('avg_acc', (np.mean(request)+slot_f1)/2, overstep[0])
#         logger.info("finishing evaluating. request_acc {}, joint_f1 {}, avg_acc {}".format(
#             np.mean(request), str(slot_f1),(np.mean(request)+slot_f1)/2))
#
#     return (np.mean(request)+slot_f1)/2
def evaluate_avg_acc(model, device,val_list, args, tokenizer, tb_writer, overstep):
    index2slot = {'O':0, 'B-weather/noun':1, 'I-weather/noun':2, 'B-location':3, 'I-location':4, 'B-datetime':5, 'I-datetime':6,
                  'B-weather/attribute':7, 'I-weather/attribute':8, 'B-reminder/todo':9, 'I-reminder/todo':10,
                  'B-alarm/alarm_modifier':11, 'B-reminder/noun':12, 'B-reminder/recurring_period':13,
                  'I-reminder/recurring_period':14, 'B-reminder/reference':15, 'I-reminder/noun':16,
                  'B-reminder/reminder_modifier':17, 'I-reminder/reference':18, 'I-reminder/reminder_modifier':19,
                  'B-weather/temperatureUnit':20, 'I-alarm/alarm_modifier':21, 'B-alarm/recurring_period':22,
                  'I-alarm/recurring_period':23}
    intent_set = {'weather/find':0, 'alarm/set_alarm':1, 'alarm/show_alarms':2, 'reminder/set_reminder':3, 'alarm/modify_alarm':4,
                  'weather/checkSunrise':5, 'weather/checkSunset':6, 'alarm/snooze_alarm':7, 'alarm/cancel_alarm':8,
                  'reminder/show_reminders':9, 'reminder/cancel_reminder':10, 'alarm/time_left_on_alarm':11}
    logger.info("start evaluating model")
    model.eval()
    # 记录tensorboardX
    loss_all = 0
    step_all = 0
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    val_dataset = MyDataset(val_list, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*7, shuffle=True, num_workers=args.num_workers,
                                collate_fn=collate_fn)
    with torch.no_grad():

        request = []
        joint = []
        slot_pre = []
        slot_gold = []
        intent_pred = []
        intent_gold = []
        slot_pred = []
        y1_list, y2_list = [], []
        for batch_idx, input_ids in enumerate(val_dataloader):
            # special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            # input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)
            inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True)
            #print('inputs_size', inputs["input_ids"].size())
            # candidate_inputs = tokenizer([candidates] * inputs['input_ids'].size(0), return_tensors="pt",
            #                             padding=True).to(device)
            inputs = inputs.to(device)
            dialogue_groundtruth = input_ids[1]
            #parameter_id = parameter_ids[input_ids[0][0].split('<nlu>')[1].split(':')[0].strip()]
            if args.local_test:
                outputs = model.generate(inputs["input_ids"], max_length=200)
            else:
                outputs = model.generate(inputs["input_ids"], max_length=200, parameter_id=0)

            for index in range(len(outputs)):
                groundtruth = dialogue_groundtruth[index].split('<|intent_slot|>')[1].split(
                    '<|endof_intent_slot|>')[0]
                #print("groundtrhtu:", groundtruth)
                try:
                    generation = tokenizer.decode(outputs[index])
                    if '<|intent_slot|>' in generation and '<|endof_intent_slot|>' in generation:
                        generation = \
                        generation.split('<|intent_slot|>')[1].split(
                            '<|endof_intent_slot|>')[0]
                    # generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]
                    else:
                        generation = 'error'
                    #print('generation:', generation)
                except IndexError:
                    generation = 'error'
                    #print("error")

                # intent_gold = ''
                # intent_pred = ''
                # joint_gold = {}
                # joint_pred = {}
                slot_values_pre = ''

                intent_gold.append(intent_set[groundtruth.split('<|slot_value|>')[0].split('<|intent|>')[1].strip()])
                slot_values_gold = groundtruth.split('<|slot_value|>')[1].strip().split()
                for slot_value in slot_values_gold:
                    # if slot_value in index2slot.keys():
                    #     slot_gold.append(index2slot[slot_value])
                    # else:
                    slot_gold.append(slot_value)

                if generation != 'error' and generation != ' ':
                    if '<|slot_value|>' in generation:

                        intent_temp = generation.split('<|slot_value|>')[0].split('<|intent|>')[1].strip()
                        if intent_temp in intent_set.keys():
                            intent_pred.append(intent_set[intent_temp])
                        else:
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
                                for x in range(len(slot_values_gold)-len(slot_values_pre)):
                                    slot_pre.append('O')
                    else:
                        intent_pred.append(len(intent_set))
                        for x in range(len(slot_values_gold)):
                            slot_pre.append('O')
                else:
                    intent_pred.append(len(intent_set))
                    for x in range(len(slot_values_gold)):
                        slot_pre.append('O')

                #print("groundtruth:", groundtruth)
                assert len(slot_pre) == len(slot_gold)

                # print("intent_pred:", intent_pred)
                # print("intent_gold:", intent_gold)

            overstep[0] += 1
            step_all += 1

            # assert len(res_pre) == len(res_gold)
            # slot_f1 = f1_score(y_true=res_gold, y_pred=res_pre,average='micro')
            # if (batch_idx % args.log_step) == 0:
            #     logger.info(
            #         "evaluate batch {} ,request_acc {}, joint_acc {}, avg_acc {}".format(
            #             batch_idx, str(intent_acc), str(slot_f1), (intent_acc+slot_f1)/2))
            # if args.local_rank == 0:
            #     tb_writer.add_scalar('avg_acc', (intent_acc+slot_f1)/2, overstep[0])
        intent_acc = accuracy_score(intent_gold, intent_pred)*100

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
            str(intent_acc), str(slot_f1),(intent_acc+slot_f1)/2))

    return (intent_acc+slot_f1)/2
def TextToDict(text, args):
    state = {}
    temp = text.strip()

    intent = temp.split('<|slot_value|>')[0].strip()
    slot = temp.split('<|slot_value|>')[1].strip()
    print("intent:", intent)
    print("slot:", slot)
    state['intent'] = intent
    state['slot'] = slot


    return state

def generate(model,tokenizer,test_list,args,device):
    logger.info('starting generating')
    save_path = open(args.save_file_name, 'w', encoding='utf-8')
    # ontology_en = json.load(open('../data/ontology/ontology_dstc2_en.json', 'r', encoding='utf-8'))
    # ontology_de = json.load(open('../data/ontology/ontology_dstc2_de.json', 'r', encoding='utf-8'))
    # ontology_it = json.load(open('../data/ontology/ontology_dstc2_it.json', 'r', encoding='utf-8'))
    # slot_match={"food": 'essen', "price range": 'preisklasse', "area": 'gegend'}
    if args.mix_train:
     input_new = open('../data/mulwoz_process/belief_de_match.txt', 'w', encoding='utf-8')

    joint_acc = 0
    model.eval()
    count = 0
    dialogue_all = []
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
    val_dataset = MyDataset(test_list)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * 10, shuffle=True,
                                num_workers=args.num_workers,
                                collate_fn=collate_fn)
    with torch.no_grad():

        request = []
        joint = []
        res_pre = []
        res_gold = []
        dialogue_all = []
        for batch_idx, input_ids in enumerate(val_dataloader):

    # for dialogue in test_list:
    #
    #     dialogue_dict = {}
    #     dialogue_dict['dialogue_idx'] = count
    #     dialogue_dict['dialogue'] = []
    #     count += 1
    #     # process dialogue
    #     dialogue_inputs = []
    #     dialogue_groundtruth = []
    #     for turns in dialogue.split('\n'):
    #         if args.prefix:
    #             temp_language = args.test_file_name.split('_')[3]
    #             print("temp_language:",temp_language)
    #             dialogue_inputs.append('<nlu> <' + temp_language + '> <'+temp_language+'>: ' + turns.split('<|intent_slot|>')[0].split('<|context|>')[1])
    #
    #         else:
    #             dialogue_inputs.append(turns.split('<|intent_slot|>')[0].split('<|context|>')[1])
    #         dialogue_groundtruth.append(turns.split('<|intent_slot|>')[1].split('<|endof_intent_slot|>')[0])
    #
    #     print("dialogue_inputs:", dialogue_inputs)
        # model generate
            dialogue_inputs  = input_ids[0]
            inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True).to(device)
            dialogue_groundtruths = input_ids[1]
            #parameter_id = parameter_ids[input_ids[0][0].split('<nlu>')[1].split(':')[0].strip()]

            pipeline = False
            if pipeline:
                outputs = []
                end_token_id = tokenizer.encode('</s>')
                print(end_token_id)
            #outputs = model.generate(inputs["input_ids"], max_length=100, forced_bos_token_id=tokenizer.encode('<en>')[0])

                for index in range(len(dialogue_inputs)):
                    inputs = tokenizer(dialogue_inputs[index], return_tensors="pt").to(device)
                    candidate_inputs = tokenizer(candidates, return_tensors="pt",
                                                 padding=True).to(device)
                    indexed_tokens = tokenizer.encode('<|belief|>')[:-1]
                    tokens_tensor = torch.tensor(indexed_tokens).to(device).unsqueeze(0)
                    #print(tokens_tensor)
                    while 1:

                        predictions = model(**inputs,decoder_input_ids=tokens_tensor)
                        candidate_outputs = model(**candidate_inputs,decoder_input_ids=tokens_tensor)

                        logits = predictions.logits
                        logits = logits + torch.mul(candidate_outputs.logits, predictions.logits)
                        predicted_index = torch.argmax(logits[0, -1, :]).item()
                        indexed_tokens += [predicted_index]
                        tokens_tensor = torch.tensor(indexed_tokens).to(device).unsqueeze(0)
                        if tokens_tensor.size(-1) > 200 or predicted_index == end_token_id[0]:
                            break
                    outputs.append(indexed_tokens)

            outputs = model.generate(inputs["input_ids"], max_length=200, parameter_id=0)

            #print(outputs)

            # tokenizer decode and
            dialogue_input_new = ''
            for index in range(len(outputs)):
                print(tokenizer.decode(outputs[index]))
                generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index]))).split('<|intent|>')[1].split('<|endof_intent_slot|>')[0]
                dialogue_groundtruth = dialogue_groundtruths[index].split('<|intent|>')[1].split('<|endof_intent_slot|>')[0]
                #generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]

                temp_list = []
                true_state = {}
                prediction = {}
                print("generation:", generation)
                print("groundtruth:", dialogue_groundtruth)
                true_state['True State'] = TextToDict(dialogue_groundtruth, args)
                prediction['Prediction'] = TextToDict(generation, args)
                temp_list.append(dialogue_inputs[index])
                temp_list.append(true_state)
                temp_list.append(prediction)

                dialogue_all.append(temp_list)

    json.dump(dialogue_all, save_path, indent=1)
    save_path.close()

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
                temp += args.root_dir_data+'/nlu_process/' + x +','
            args.train_file_name = temp[:-1]
        else:
            args.train_file_name = args.root_dir_data+'/nlu_process/' + args.train_file_name
        args.val_file_name = args.root_dir_data+'/nlu_process/' + args.val_file_name
        args.test_file_name = args.root_dir_data+'/nlu_process/' + args.test_file_name
        args.ontology_path = args.root_dir_data+'/ontology/ontology_dstc2_%s.json' % args.target_language

        return tb_writer, multi_gpu

def evaluate_result(evaluated_dialogues, language):
    intent = []
    slot = []
    index2slot = {'O': 0, 'B-weather/noun': 1, 'I-weather/noun': 2, 'B-location': 3, 'I-location': 4, 'B-datetime': 5,
                  'I-datetime': 6,
                  'B-weather/attribute': 7, 'I-weather/attribute': 8, 'B-reminder/todo': 9, 'I-reminder/todo': 10,
                  'B-alarm/alarm_modifier': 11, 'B-reminder/noun': 12, 'B-reminder/recurring_period': 13,
                  'I-reminder/recurring_period': 14, 'B-reminder/reference': 15, 'I-reminder/noun': 16,
                  'B-reminder/reminder_modifier': 17, 'I-reminder/reference': 18, 'I-reminder/reminder_modifier': 19,
                  'B-weather/temperatureUnit': 20, 'I-alarm/alarm_modifier': 21, 'B-alarm/recurring_period': 22,
                  'I-alarm/recurring_period': 23}

    slot_gold_all = []
    slot_pre_all = []
    for turn in evaluated_dialogues:
        True_state = turn[1]['True State']
        Prediction = turn[2]['Prediction']
        intent_gold = set(True_state['intent'])
        intent_pre = set(Prediction['intent'])
        slot_gold = True_state['slot'].split()
        slot_pre = Prediction['slot'].split()

        for slot_value in slot_pre:
            if slot_value in index2slot.keys():
                slot_pre_all.append(index2slot[slot_value])
            else:
                slot_pre_all.append(len(index2slot))

        for slot_value in slot_gold:
            if slot_value in index2slot.keys():
                slot_gold_all.append(index2slot[slot_value])
            else:
                slot_gold_all.append(len(index2slot))

        if len(slot_gold) < len(slot_pre):
            slot_gold_all += ([len(index2slot)]*(len(slot_pre)-len(slot_gold)))
        elif len(slot_gold) > len(slot_pre):
            slot_pre_all += ([len(index2slot)] * (len(slot_gold) - len(slot_pre)))

        intent.append(intent_gold == intent_pre)

    assert len(slot_pre_all) == len(slot_gold_all)
    slot_f1 = f1_score(y_true=slot_gold_all, y_pred=slot_pre_all, average='micro')


            #slot.append(slot_gold == slot_pre)

    logger.info("final test intent_acc:" + str(np.mean(intent)))
    logger.info("final test slot_acc:" + str(slot_f1))

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
    if args.cuda:
    # device = torch.device("cuda", args.local_rank)

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
    if args.local_test:
        tokenizer = MT5Tokenizer.from_pretrained("../../mt5/")
    else:
        tokenizer = MT5Tokenizer.from_pretrained("../../GPT2-chitchat/model/mt5/")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|system|>','<en>', '<nlu>','<de>','<it>'
                                                                  ,'<|number|>','<|intent_slot|>' ,'<|endof_intent_slot|>'
                                                                ,'<|slot_value|>']})

    #print(os.path.abspath(__file__))
    #r = open('G:/learn/code/Multilingual_Dialogue/data/nlu_process/nlubiomltcrossth_en_train.txt', 'r')

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
    if args.cuda:
       model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
    # 加载数据
    logger.info("loading traing data")
    # consider train_path list
    train_list = []
    if len(args.train_file_name.split(',')) == 1:
         train_temp = open(args.train_file_name, "r", encoding='utf-8').read().split('\n\n')[0:-1]
         temp_language = args.train_file_name.split('_')[3]
         target_language = args.target_language
         for x in train_temp:
             if args.prefix:
                for y in x.split('\n'):
                    #train_list.append('<|endoftext|> <|context|> ' + '<nlu> <' + temp_language + '> <' + temp_language + '>: ' + y.split('<|context|>')[1])
                    if args.cross:
                        train_list.append(
                            '<|endoftext|> <|context|> ' + '<nlu> <' + target_language + '> <' + target_language + '>: ' +
                            y.split('<|context|>')[1])
                    else:
                        train_list.append(
                            '<|endoftext|> <|context|> ' + '<nlu> <' + temp_language + '> <' + temp_language + '>: ' +
                            y.split('<|context|>')[1])
             else:
                train_list += x.split('\n')
    else:
         train_list = []
         for x in args.train_file_name.split(','):
             temp_language = x.split('_')[3]
             target_language = args.target_language
             print("train_temp_language:", temp_language)
             train_temp = open(x, "r", encoding='utf-8').read().split('\n\n')[0:-1]
             for y in train_temp:
                 for z in y.split('\n'):
                     if args.cross:
                         train_list.append(
                             '<|endoftext|> <|context|> ' + '<nlu> <' + target_language + '> <' + target_language + '>: ' +
                             z.split('<|context|>')[1])
                     else:
                         train_list.append(
                             '<|endoftext|> <|context|> ' + '<nlu> <' + temp_language + '> <' + temp_language + '>: ' +
                             z.split('<|context|>')[1])
    print('train length:', len(train_list))
    logger.info("loading val data")
    val_temp = open(args.val_file_name, "r", encoding='utf-8').read().split('\n\n')[0:-1]
    val_list = []
    temp_language = args.val_file_name.split('_')[3]
    CURRENT_LANG = temp_language
    print("dev_temp_language:", temp_language)
    for data in val_temp:
        if args.prefix:
            try:
                for x in data.split('\n'):
                    val_list.append('<|endoftext|> <|context|> ' + '<nlu> <' + temp_language + '> <' + temp_language + '>: ' +x.split('<|context|>')[1])
            except IndexError:
                print(x)
        else:
            val_list += data.split('\n')
    print('val length:', len(val_list))
    logger.info("loading testing data")
    test_temp = open(args.test_file_name, "r", encoding='utf-8').read().split('\n\n')[0:-1]
    test_list = []
    temp_language = args.test_file_name.split('_')[3]
    print("test_temp_language:", temp_language)
    for data in test_temp:
        if args.prefix:
            try:
                for x in data.split('\n'):
                    test_list.append(
                        '<|endoftext|> <|context|> ' + '<nlu> <' + temp_language + '> <' + temp_language + '>: ' +
                        x.split('<|context|>')[1])
            except IndexError:
                print(x)
        else:
            test_list += data.split('\n')
    print('test length:', len(test_list))
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
            overstep = [0]
            # min_model = '../output/model/MT5_mot_belieturn_en_model/model_epoch46/'
            model = MT5ForConditionalGeneration.from_pretrained(best_model)
            model.to(device)
            result = evaluate_avg_acc(model, device, test_list, args, tokenizer, tb_writer, overstep)
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
                #model = MT5ForConditionalGeneration.from_pretrained('../output/model/nlu_cross_th_biomlt_model/model_epoch40/')
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
        overstep = [0]
        model.to(device)

        generate(model, tokenizer, test_list, args, device)
        evaluated_dialogues = json.load(open(args.save_file_name, 'r', encoding='utf-8'))
        test_language = args.test_file_name.split('_')[-2]
        evaluate_result(evaluated_dialogues, test_language)

        result = evaluate_avg_acc(model, device, test_list, args, tokenizer, tb_writer, overstep)
    else:
        if args.local_rank == 0:
            overstep = [0]
            #min_model = '../output/model/MT5_mot_belieturn_en_model/model_epoch46/'

            generate(model, tokenizer, test_list, args, device)
            evaluated_dialogues = json.load(open(args.save_file_name, 'r', encoding='utf-8'))
            test_language = args.test_file_name.split('_')[-2]
            evaluate_result(evaluated_dialogues, test_language)

            model = MT5ForConditionalGeneration.from_pretrained(args.pretrained_model)
            model.to(device)
            result = evaluate_avg_acc(model, device, test_list, args, tokenizer, tb_writer, overstep)
            #generate(model, tokenizer, test_list, args, device)
            #evaluated_dialogues = json.load(open(args.save_file_name, 'r', encoding='utf-8'))
            #test_language = args.test_file_name.split('_')[-2]
            #evaluate_result(evaluated_dialogues, test_language)
if __name__ == '__main__':
    main()
