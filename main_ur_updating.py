import random

import torch
import numpy as np
import os

from time import time
from prettytable import PrettyTable

from parser import parse_args
from data_loader import load_data
from MMGUP import Recommender
from evaluate import test
from helper import early_stopping
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):

    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    print('======================================')
    args = parse_args()
    print(args.out_dir)
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, tune_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    tune_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in tune_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    best_epoch = 0
    stopping_step = 0
    should_stop = False
    
    model.update_user_region_embedding()
    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'])
            batch_loss, _, _, batch_cor = model(batch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            cor_loss += batch_cor
            s += args.batch_size

        train_e_t = time()
        
        model.update_user_region_embedding()
        if epoch % 10 == 9 or epoch == 1:
#             model.update_user_region_embedding()
            """tuning"""
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            #train_res = PrettyTable()
            #train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio", "AUC", "logloss"]
            #train_res.add_row(
            #    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio'], ret['auc'], ret['logloss']])                 
            #print('Tuing performance:')
            #print(train_res)
            
            
            """testing"""
            ret_test = test(model, user_dict, n_params, mode='test')
            print('====================Testing performance @ epoch %d=======================' % epoch)
            print(ret_test['recall'], ret_test['ndcg'], ret_test['precision'], ret_test['hit_ratio'])

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][3], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=5)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][3] == cur_best_pre_0 and args.save:
                best_epoch = epoch
                if not os.path.exists(args.out_dir):
                    os.makedirs(args.out_dir)
                torch.save(model.state_dict(), args.out_dir + '/model_' + args.dataset + '.ckpt')

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_e_t - train_s_t, epoch, #loss.item(), cor_loss.item()))
loss.item(), cor_loss))

    print('early stopping at %d, tuning AUC:%.4f' % (epoch, cur_best_pre_0))
    print('testing performance')
    print(ret_test['recall'], ret_test['ndcg'], ret_test['precision'], ret_test['hit_ratio'])
    print('best epoch %d' % best_epoch)
