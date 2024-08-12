import pdb
import logging
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy

class Tester(object):
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model
        # self.model_mf = args.model_mf
        self.history_dic = dataloader.historical_dict
        self.history_csr = dataloader.train_csr
        self.dataloader = dataloader.dataloader_test
        self.test_dic = dataloader.test_dic
        self.cate = dataloader.item_category_dic #np.array(list(dataloader.category_dic.values()))
        self.metrics = args.metrics
        #self.pcu = dataloader.pcu

    def judge(self, users, items):

        results = {metric: 0.0 for metric in self.metrics}
        # for ground truth test
        # items = self.ground_truth_filter(users, items)
        stat = self.stat(items)
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            for i in range(len(items)):
                results[metric] += f(items[i], test_pos = self.test_dic[users[i]], num_test_pos = len(self.test_dic[users[i]]), count = stat[i], model = self.model, category = [self.cate[item].tolist() for item in items[i].tolist()])
        return results

    def ground_truth_filter(self, users, items):
        batch_size, k = items.shape
        res = []
        for i in range(len(users)):
            gt_number = len(self.test_dic[users[i]])
            if gt_number >= k:
                res.append(items[i])
            else:
                res.append(items[i][:gt_number])
        return res

    def test(self):
        results = {}
        h = self.model.get_embedding()
        count = 0

        for k in self.args.k_list:
            results[k] = {metric: 0.0 for metric in self.metrics}

        for batch in tqdm(self.dataloader):
            users = batch[0]
            count += users.shape[0]
            # count += len(users)
            scores = self.model.get_score(h, users)

            # test ground truth
            # scores_ls = []
            # num_item = scores.shape[1]
            # for user in users:
            #     score_user = torch.zeros(num_item, device = scores.device)
            #     gt = torch.tensor(self.test_dic[user], device = scores.device)
            #     score_user[gt] = 1.0
            #     scores_ls.append(score_user)
            # scores = torch.stack(scores_ls)

            users = users.tolist()
            mask = torch.tensor(self.history_csr[users].todense(), device = scores.device).bool()
            scores[mask] = -float('inf')

            _, recommended_items = torch.topk(scores, k = max(self.args.k_list))
            recommended_items = recommended_items.cpu()
            for k in self.args.k_list:

                results_batch = self.judge(users, recommended_items[:, :k])

                for metric in self.metrics:
                    results[k][metric] += results_batch[metric]

        for k in self.args.k_list:
            for metric in self.metrics:
                results[k][metric] = results[k][metric] / count
        self.show_results(results)

    def show_results(self, results):
        for metric in self.metrics:
            for k in self.args.k_list:
                logging.info('For top{}, metric {} = {}'.format(k, metric, results[k][metric]))

    def stat(self, items):
        stat = []
        for item in items:
            item_c = []
            for itemK in item:
                item_c += self.cate[itemK.item()].tolist()
            stat.append(np.unique(np.array(item_c), return_counts=True)[1])
        #stat = [np.unique(self.cate[item], return_counts=True)[1] for item in items]
        return stat


class Metrics(object):

    def __init__(self):
        pass

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'ndcg': Metrics.ndcg,
            'coverage': Metrics.coverage,
            'entropy': Metrics.entropy,
            'gini': Metrics.gini
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def coverage(items, **kwargs):

        count = kwargs['count']

        return count.size
    
    @staticmethod
    def ndcg(items, **kwargs):
        test_pos = kwargs['test_pos']
        hit = np.isin(items, test_pos[:len(items)])
        ranks = np.arange(1, len(items)+1)
        dcg = hit / np.log2(ranks+1.0)
        idcg = 1.0 / np.log2(ranks+1.0)
        return np.sum(dcg) / np.sum(idcg)

    @staticmethod
    def andcg(items, **kwargs):
        pcu = kwargs['pcu']
        alpha = 0.5
        test_pos = kwargs['test_pos']
        cates = kwargs['category']
        hit = np.isin(items, test_pos[:len(items)])
        adcg = 0
        aidcg = 0
        rc = {}
        for i in range(len(items)):
            tmp = 0
            for cate in cates[i]:
                if cate not in rc:
                    rc[cate] = 0
                #pcu_cate = 1e-5 if cate not in pcu else pcu[cate]
                tmp += np.power(1-alpha, rc[cate]) * hit[i] #pcu_cate
                rc[cate] += 1
            adcg += tmp/np.log2(i+2)

        rc = {}
        for i in range(len(test_pos[:len(items)])):
            tmp = 0
            for cate in cates[i]:
                if cate not in rc:
                    rc[cate] = 0
                #pcu_cate = 1e-5 if cate not in pcu else pcu[cate]
                tmp += np.power(1-alpha, rc[cate]) #* pcu_cate
                rc[cate] += 1
            aidcg += tmp/np.log2(i+2)
        return adcg / aidcg

    @staticmethod
    def entropy(items, **kwargs):
        #num_test_pos = kwargs['num_test_pos']
        #cates = sum(kwargs['category'], [])
        #p = {}
        #for cate in cates:
        #    if cate not in p:
        #        p[cate] = 1
        #    else:
        #        p[cate] += 1
        #entropy = [p[key]/len(items)*np.log2(p[key]/len(items)) for key in p]
        #return -np.sum(entropy)
        count = kwargs['count']
        return entropy(count)

    @staticmethod
    def gini(items, **kwargs):
        #num_test_pos = kwargs['num_test_pos']
        #cates = sum(kwargs['category'], [])
        #p = {}
        #for cate in cates:
        #    if cate not in p:
        #        p[cate] = 1
        #    else:
        #        p[cate] += 1
        count = kwargs['count']
        #unique_count = count / len(items)
        #gini = np.sum(abs(unique_count-unique_count.reshape(-1,1))) / unique_count.shape[0] ** 2
        #return gini.item()
        count = np.sort(count)
        n = len(count)
        cum_count = np.cumsum(count)

        return (n + 1 - 2 * np.sum(cum_count) / cum_count[-1]) / n
