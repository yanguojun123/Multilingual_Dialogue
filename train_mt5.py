from torch.backends import cudnn
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from os.path import join, exists
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import shutil
import re
PAD = '[PAD]'
pad_id = 0
logger = None
CODE_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import time
def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    #parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
    #                   help='选择模型参数')
    #parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_raw_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--raw', action='store_true', help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=60, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=4, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--seed', type=int, default=12345, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--train_mmi', action='store_true', help="若指定该参数，则训练DialoGPT的MMI模型")
    parser.add_argument('--train_mmi_tokenized_path', default='data/train_mmi_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料的每段对话翻转，然后进行tokenize之后的数据的存放位置，用于训练MMI模型')
    parser.add_argument('--mmi_model_output_path', default='mmi_model', type=str, required=False, help='MMI模型保存路径')
    parser.add_argument('--eval_all_checkpoints', action='store_true', help='在所有模型上评价')
    parser.add_argument('--train_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--val_path', default='data/val.txt', type=str, required=False, help='原始验证语料')
    parser.add_argument('--test_path', default='data/test.txt', type=str, required=False, help='原始测试语料')
    parser.add_argument('--save_path', default='output/pretrained_mt5.txt', type=str, required=False, help='保存生成结果的路径')
    parser.add_argument('--ontology_path', default='ontology/ontology_dstc2_en.json', type=str, required=False, help='存放slot定义的路径')
    parser.add_argument('--language', default='en', type=str, required=False, help='language')
    parser.add_argument("--local_rank", type=int, default=-1, help='distributed')
    parser.add_argument('--root_dir_data', default='%s/data' % CODE_ROOT_DIR, type=str, required=False,
                        help='Root directory of all data，e.g., dataset, vocab, pretrained models')
    parser.add_argument('--root_dir_output', default='%s/output' % CODE_ROOT_DIR, type=str, required=False,
                        help='Root directory of all output，e.g., model, result')
    parser.add_argument('--continue_epoch', default=0, type=int, required=False, help='set start epoch')
    parser.add_argument('--prefix', action='store_true', required=False, help='train with prefix')

    # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
    # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
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

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

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
        inputs.append(batch[btc_idx].split("<|endofcontext|>")[0].split("<|context|>")[1])
        labels.append(batch[btc_idx].split("<|endofbelief|>")[0].split("<|belief|>")[1])
    return [inputs, labels]

def train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer):
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
        overall_step = 0
        if torch.cuda.is_available():
            sampler = DistributedSampler(train_dataset)
            sampler.set_epoch(epoch)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      collate_fn=collate_fn, sampler=sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers,
                                          collate_fn=collate_fn)
        epoch_start_time = datetime.now()
        for batch_idx, input_ids in enumerate(train_dataloader):

            inputs = tokenizer(input_ids[0], return_tensors="pt",padding=True).to(device)
            with tokenizer.as_target_tokenizer():
                if input_ids[1] is not None:
                    labels = tokenizer(input_ids[1], return_tensors="pt", padding=True).to(device)
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                outputs = model(**inputs, labels=labels["input_ids"])
                loss = outputs.loss
                #print("loss:", loss)
                if multi_gpu:
                    loss = loss.mean()
                if args.gradient_accumulation > 1:
                    loss = loss.mean() / args.gradient_accumulation
                #loss.backward(loss.clone().detach())
                loss.backward(retain_graph=True)
                '''if temp1 != model.state_dict()['module.lm_head.weight'][0, 10]:
                    print("Parameter-1 have updated!")
                if temp2 != model.state_dict()['module.lm_head.weight'][250101, 10]:
                    print("Parameter-2 have updated!")'''
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
            if args.train_mmi:  # 当前训练MMI模型
                model_path = join(args.mmi_model_output_path, 'model_epoch{}'.format(epoch + 1))
            else:  # 当前训练对话模型
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


def evaluate(model, device, val_list, multi_gpu, args, tokenizer, tb_writer, overstep):
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    # 记录tensorboardX
    loss_all = 0
    accuracy_epoch = 0
    # tb_writer = SummaryWriter(log_dir=args.writer_dir)
    val_dataset = MyDataset(val_list)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=collate_fn)
    with torch.no_grad():

        for batch_idx, input_ids in enumerate(val_dataloader):
            #special_index = (input_ids[:, 0:6] - np.ones((input_ids.size(0), 6), dtype=int)).to(device)
            #input_ids = input_ids[:, 6:].to(device)
            # input_ids.to(device)
            inputs = tokenizer(input_ids[0], return_tensors="pt", padding=True)
            inputs = inputs.to(device)

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(input_ids[1], return_tensors="pt", padding=True)
                labels = labels.to(device)
                outputs = model(**inputs, labels=labels["input_ids"])
                loss = outputs.loss
                loss_all += loss.mean()
                overstep[0] += 1

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
            loss_all / overstep[0]))

    return loss

def TextToDict(text,args):
    state = {}
    if args.language == 'en':
        state = {'food':'none', 'price range':'none', 'request':[], 'area':'none'}
    if args.language == 'de':
        state = {'essen': 'none', 'preisklasse': 'none', 'request': [], 'gegend': 'none'}
    if args.language == 'it':
        state = {'cibo': 'none', 'prezzo': 'none', 'request': [], 'area': 'none'}
    temp = text.strip()
    for slots in temp.split(','):
        if 'request' in slots:
            state['request'].append(slots.split()[-1])
        else:
            for slot in ['food', 'price range', 'area', 'essen', 'preisklasse', 'gegend', 'cibo', 'prezzo']:
                if slot in slots:
                    state[''+slot] = slots.split()[-1]
    state['request'] = list(set(state['request']))
    return state
def generate(model,tokenizer,test_list,args,device):
    logger.info('starting generating')
    save_path = open(args.save_path, 'w', encoding='utf-8')
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn1)
    joint_acc = 0
    count = 0
    dialogue_all = []
    for dialogue in test_list:

        dialogue_dict = {}
        dialogue_dict['dialogue_idx'] = count
        dialogue_dict['dialogue'] = []
        count += 1
        # process dialogue
        dialogue_inputs = []
        dialogue_groundtruth = []
        for turns in dialogue.split('\n'):
            if args.prefix:
                dialogue_inputs.append('<dst> <'+args.language+'>: ' + turns.split('<|endofcontext|>')[0].split('<|context|>')[1])
            else:
                dialogue_inputs.append(turns.split('<|endofcontext|>')[0].split('<|context|>')[1])
            dialogue_groundtruth.append(turns.split('<|belief|>')[1].split('<|endofbelief|>')[0])

        # model generate

        inputs = tokenizer(dialogue_inputs, return_tensors="pt", padding=True).to(device)
        #print(inputs)
        #outputs = model.generate(inputs["input_ids"], max_length=100, forced_bos_token_id=tokenizer.encode('<en>')[0])
        outputs = model.generate(inputs["input_ids"], max_length=100)
        #print(outputs)

        # tokenizer decode and
        for index in range(len(outputs)):
            print(tokenizer.decode(outputs[index]))
            generation = re.sub('</s>', '', re.sub('<pad>', '', tokenizer.decode(outputs[index])))
            #generation = tokenizer.decode(outputs[index]).split('</s>')[0].split('<pad>')[1]

            temp_list = []
            true_state = {}
            prediction = {}
            true_state['True State'] = TextToDict(dialogue_groundtruth[index], args)
            prediction['Prediction'] = TextToDict(generation, args)
            temp_list.append(dialogue_inputs[index])
            temp_list.append(true_state)
            temp_list.append(prediction)

            dialogue_dict['dialogue'].append(temp_list)
        dialogue_all.append(dialogue_dict)
    json.dump(dialogue_all, save_path, indent=1)
    save_path.close()

def init(args):

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
        if len(args.train_path.split(','))>1:
            temp = ''
            for x in args.train_path.split(','):
                temp += '../data/mulwoz_process/' + x +','
            args.train_path = temp[:-1]
        else:
            args.train_path = '../data/mulwoz_process/' + args.train_path
        args.dev_path = '../data/mulwoz_process/' + args.dev_path
        args.test_path = '../data/mulwoz_process/' + args.test_path

        args.log_path = '../result/' + args.log_path
        args.dialogue_model_output_path = '../model/' + args.dialogue_model_output_path
        args.save_path = '../result/' + args.save_path
        return tb_writer, multi_gpu
def main():
    args = setup_train_args()

    tb_writer = ''
    multi_gpu = ''
    # 日志同时输出到文件和console
    global logger
    logger = create_logger(args)
    print("args:", args)
    # Setup CUDA, GPU & distributed training
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'

    device = torch.device("cuda", args.local_rank)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    logger.info('using device:{}'.format(device))
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    # 当得到比较好的结果时我们通常希望这个结果是可以复现
    if args.seed:
        set_random_seed(args)

    # 设置使用哪些显卡进行训练
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # tokenizer的字典大小
    model = MT5ForConditionalGeneration.from_pretrained("../GPT2-chitchat/model/mt5/")
    tokenizer = MT5Tokenizer.from_pretrained("../GPT2-chitchat/model/mt5/")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|system|>','<en>', '<dst>','<de>','<it>']})
    model_dict = model.state_dict()
    for key in model_dict.keys():
        print('key: '+key, model_dict[''+key].size())
    pe = model_dict['encoder.embed_tokens.weight'][250100:, :]
    sw = model_dict['shared.weight'][250100:, :]
    #pe.require_grad = False
    #sw.require_grad = False


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

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # 加载数据
    logger.info("loading traing data")
    # consider train_path list
    train_list = []
    if len(args.train_path.split(',')) == 1:
         train_temp = open(args.train_path, "r", encoding='utf-8').read().split('\n\n')[0:-1]
         temp_language = args.train_path.split('_')[1]
         for x in train_temp:
             if args.prefix:
                for y in x.split('\n'):
                    train_list.append(
                     '<|endoftext|> <|context|> ' + '<dst> <' + temp_language + '>: ' + y.split('<|context|>')[1])
             else:
                train_list += x.split('\n')
    else:
         for x in args.train_path.split(','):
             temp_language = x.split('_')[1]
             train_temp = open(x, "r", encoding='utf-8').read().split('\n\n')[0:-1]
             for y in train_temp:
                 for z in y.split('\n'):
                    train_list.append('<|endoftext|> <|context|> ' + '<dst> <'+temp_language+'>: ' + z.split('<|context|>')[1])

    logger.info("loading val data")
    val_temp = open(args.val_path, "r", encoding='utf-8').read().split('\n\n')[0:-1]
    val_list = []
    temp_language = args.val_path.split('_')[1]
    for data in val_temp:
        if args.prefix:
            for x in data.split('\n'):
                val_list.append('<|endoftext|> <|context|> ' + '<dst> <'+temp_language+'>: ' + x.split('<|context|>')[1])
        else:
            val_list += data.split('\n')
    logger.info("loading testing data")
    test_list = open(args.test_path, "r", encoding='utf-8').read().split('\n\n')[0:-1]

    # 开始训练

    train(model, device, train_list, multi_gpu, args, tokenizer, tb_writer)
    # 模型验证
    if args.local_rank == 0:
        min_model = ''
        if args.eval_all_checkpoints:
            checkpoints = [args.dialogue_model_output_path + c for c in
                           sorted(os.listdir(args.dialogue_model_output_path))[1:-1]]
            logger.info("Evaluate the following checkpoints: {}".format(checkpoints))
            overstep = [0]
            min_loss = 100000

            for x in range(1, args.epochs + 1):
                checkpoint = args.dialogue_model_output_path + 'model_epoch' + str(x)
                model = MT5ForConditionalGeneration.from_pretrained(checkpoint)
                logger.info("Evaluate the checkpoint: {}".format(checkpoint))
                model.resize_token_embeddings(vocab_size)
                model.to(device)
                result = evaluate(model, device, val_list, multi_gpu, args, tokenizer, tb_writer, overstep)
                if result < min_loss:
                    min_loss = result
                    min_model = checkpoint
            logger.info("the best model is " + min_model)
        tb_writer.close()
        #min_model = 'model/pretrained_mbart_en_test21_model/model_epoch48/'
        model = MT5ForConditionalGeneration.from_pretrained(min_model)
        model.to(device)
        generate(model, tokenizer, test_list, args, device)
if __name__ == '__main__':
    main()
