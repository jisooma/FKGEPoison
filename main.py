import torch
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os

from fede import FedE
import numpy as np
import datetime


def init_dir(args):
    # state
    args.state_dir = args.state_dir
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)
 
    # tensorboard log
    args.tb_log_dir = args.tb_log_dir
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    args.log_dir = args.log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def init_logger(args):
    log_file = os.path.join(args.log_dir,
                            args.dataset_name + '_' + args.client_model + '_' +str(args.num_client)+'_'+ str(args.attack_entity_ratio) + '.log')

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='a+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    from set_args import args
    from DPA_S import DPA_S
    from FMPA_S import FMPA_S
    from CPA import CPA

    import warnings
    warnings.filterwarnings('ignore')

    # random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # init dir, logger and log args
    init_dir(args)
    init_logger(args)
    args_str = json.dumps(vars(args))
    logging.info(args_str)

    # assign cuda device
    args.gpu = torch.device(args.gpu)

    # init tensorboard
    writer = SummaryWriter(os.path.join(args.tb_log_dir, args.dataset_name
                                        + '_' + args.client_model +'_'+str(args.num_client)+ '_' + str(args.attack_entity_ratio)))
    args.writer = writer

    args.name = args.dataset_name + '_' + args.client_model + '_' + str(args.num_client) + "_" + str(args.attack_entity_ratio)
    print('***************', args.setting)
    client = args.num_client
    dataset_name = args.dataset_name

    with open("../process_data/client_data/" + dataset_name + "_" + str(client) + "_with_new_id.json", "r") as f:
        all_data = json.load(f)

    # FedE
    if args.setting == 'FedE':
        learner = FedE(args, all_data)
        if args.mode == 'train':
            learner.train()
        elif args.mode == 'test':
            learner.before_test_load()
            learner.evaluate(istest=True)
            results = learner.evaluate(ispoisoned=True)

    # DPA_S: Dynamic Poisoning Attack
    if args.setting == 'DPA_S':
        learner = DPA_S(args, all_data)
        if args.mode == 'train':
            learner.train()
        elif args.mode == 'test':
            learner.before_test_load()
            learner.evaluate(istest=True)
            results = learner.evaluate(ispoisoned=True)

    # FMPA-S: Fixed Model Poisoning Attack
    if args.setting == 'FMPA_S':
        learner = FMPA_S(args, all_data)
        if args.mode == 'train':
            learner.train()
        elif args.mode == 'test':
            learner.before_test_load()
            learner.evaluate(istest=True)
            results = learner.evaluate(ispoisoned=True)

    # CPA: Client Poisoning Attack
    if args.setting == 'CPA':
        learner = CPA(args, all_data)
        if args.mode == 'train':
            learner.train()
        elif args.mode == 'test':
            learner.before_test_load()
            learner.evaluate(istest=True)
            results = learner.evaluate(ispoisoned=True)
