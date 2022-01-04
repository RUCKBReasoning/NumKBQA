import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
import os, math
from NSM.train.init import init_nsm
from NSM.train.evaluate_nsm import Evaluator_nsm
from NSM.data.load_data_super import load_data
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch.optim as optim
tqdm.monitor_iterval = 0


def calc_type_accuracy(question_type_list, hits_list):
    assert len(question_type_list) == len(hits_list)
    num_data = len(hits_list)
    q_tot = {}
    q_hit = {}
    q_set = set(question_type_list)
    for type in q_set:
        q_tot[type] = 0
        q_hit[type] = 0
    for i in range(num_data):
        q_type = question_type_list[i]
        q_tot[q_type] += 1
        if hits_list[i] > 0:
            q_hit[q_type] += 1
    res_str = ""
    for type in q_tot:
        res_str += "{} type question,  {} cases among {} cases are right, accuracy: {:.4f}, ".format(type, q_hit[type],
                                                                                                     q_tot[type],
                                                                         float(q_hit[type]) / float(q_tot[type]))
    return res_str[:-2], float(q_hit["superlative"]) / float(q_tot["superlative"])


class Trainer_KBQA(object):
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.best_h1 = 0.0
        self.best_f1 = 0.0
        self.eps = args['eps']
        self.learning_rate = self.args['lr']
        self.test_batch_size = args['test_batch_size']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.train_kl = args['train_KL']
        self.num_step = args['num_step']
        self.use_label = args['use_label']
        self.use_num = args['use_num']
        self.reset_time = 0
        self.load_data(args)
        if 'decay_rate' in args:
            self.decay_rate = args['decay_rate']
        else:
            self.decay_rate = 0.98
        self.mode = "teacher"
        self.model_name = self.args['model_name']
        self.student = init_nsm(self.args, self.logger, len(self.entity2id), self.num_kb_relation,
                                  len(self.word2id))
        self.student.to(self.device)
        self.evaluator = Evaluator_nsm(args=args, student=self.student, entity2id=self.entity2id,
                                       relation2id=self.relation2id, device=self.device)
        if not args['is_eval']:
            self.load_pretrain()
        self.optim_def()

    def optim_def(self):
        trainable = filter(lambda p: p.requires_grad, self.student.parameters())
        self.optim_student = optim.Adam(trainable, lr=self.learning_rate)
        if self.decay_rate > 0:
            self.scheduler = ExponentialLR(self.optim_student, self.decay_rate)

    def load_data(self, args):
        dataset = load_data(args)
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        self.entity2id = dataset["entity2id"]
        self.relation2id = dataset["relation2id"]
        self.word2id = dataset["word2id"]
        self.num_kb_relation = self.test_data.num_kb_relation
        self.num_entity = len(self.entity2id)

    def load_pretrain(self):
        args = self.args
        if args['load_num'] is not None and args['use_num']:
            ckpt_path = os.path.join(args['checkpoint_dir'], args['load_num'])
            print("Load ckpt from", ckpt_path)
            self.load_num_ckpt(ckpt_path)
        if args['load_experiment'] is not None:
            ckpt_path = os.path.join(args['checkpoint_dir'], args['load_experiment'])
            print("Load ckpt from", ckpt_path)
            self.load_ckpt(ckpt_path)

    def evaluate(self, data, test_batch_size=20, mode="teacher", write_info=False):
        return self.evaluator.evaluate(data, test_batch_size, write_info)

    def train(self, start_epoch, end_epoch):
        eval_every = self.args['eval_every']
        _, _, _, eval_superlative_acc = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
        test_f1, test_h1, res_str, test_superlative_acc = self.evaluate(self.test_data, self.test_batch_size, mode="teacher")
        self.logger.info("initial TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_h1))
        self.logger.info("initial superlative acc: {:.4f}".format(test_superlative_acc))
        print("Strat Training------------------")
        if "webqsp" in self.args['data_folder']:
            self.best_superlative_acc = 0.0
        else:
            self.best_superlative_acc = eval_superlative_acc
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()
            if self.decay_rate > 0:
                self.scheduler.step()
            self.logger.info("Epoch: {}, loss : {:.4f}, time: {}".format(epoch + 1, loss, time.time() - st))
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(np.mean(h1_list_all), np.mean(f1_list_all)))
            if self.use_num:
                res_str, train_superlative_acc = calc_type_accuracy(self.question_type_list, hits_list=h1_list_all)
                self.logger.info("Training type hits: {}".format(res_str))
            if (epoch + 1) % eval_every == 0 and epoch + 1 > 0:
                if self.model_name == "back":
                    eval_f1 = np.mean(f1_list_all)
                    eval_h1 = np.mean(h1_list_all)
                else:
                    eval_f1, eval_h1, res_str, eval_superlative_acc = self.evaluate(self.valid_data,
                                                                                    self.test_batch_size,
                                                                                    mode="teacher")
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                if self.use_num:
                    self.logger.info(res_str)
                if eval_h1 > self.best_h1:
                    self.best_h1 = eval_h1
                    self.save_ckpt("h1")
                if eval_f1 > self.best_f1:
                    self.best_f1 = eval_f1
                    self.save_ckpt("f1")
                if self.use_num:
                    if "webqsp" in self.args['data_folder']:
                        if train_superlative_acc > self.best_superlative_acc:
                            self.best_superlative_acc = train_superlative_acc
                            self.save_ckpt("superlative-train-acc")
                            self.logger.info("Best training superlative acc: {:.3f}".format(train_superlative_acc))
                    else:
                        if eval_superlative_acc > self.best_superlative_acc:
                            self.best_superlative_acc = eval_superlative_acc
                            self.save_ckpt("superlative-dev-acc")
                            self.logger.info("Best dev superlative acc: {:.3f}".format(eval_superlative_acc))
                test_f1, test_h1, res_str, test_superlative_acc = self.evaluate(self.test_data, self.test_batch_size,
                                                                                mode="teacher")
                self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_h1))
                self.logger.info(res_str)

        self.save_ckpt("final")
        self.logger.info('Train Done! Evaluate on testset with saved model')
        print("End Training------------------")
        if self.model_name != "back":
            self.evaluate_best(self.mode)

    def evaluate_best(self, mode):
        if self.use_num:
            if "webqsp" in self.args['data_folder']:
                filename = os.path.join(self.args['checkpoint_dir'],
                                        "{}-superlative-train-acc.ckpt".format(self.args['experiment_name']))
                self.logger.info("Best superlative train acc evaluation")
            else:
                filename = os.path.join(self.args['checkpoint_dir'],
                                        "{}-superlative-dev-acc.ckpt".format(self.args['experiment_name']))
                self.logger.info("Best superlative dev acc evaluation")
            self.load_ckpt(filename)
            eval_f1, eval_h1, res_str, acc = self.evaluate(self.test_data, self.test_batch_size, mode="teacher",
                                                      write_info=False)
            self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
            self.logger.info(res_str)

        filename = os.path.join(self.args['checkpoint_dir'], "{}-h1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, res_str, acc = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Best h1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
        self.logger.info(res_str)

        filename = os.path.join(self.args['checkpoint_dir'], "{}-f1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, res_str, acc = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Best f1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
        self.logger.info(res_str)

        filename = os.path.join(self.args['checkpoint_dir'], "{}-final.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, res_str, acc = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Final evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
        self.logger.info(res_str)

    def evaluate_single(self, filename):
        if filename is not None:
            self.load_ckpt(filename)
        test_f1, test_hits, res_str, acc = self.evaluate(self.test_data, self.test_batch_size, mode="teacher",
                                                         write_info=True)
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_hits))
        self.logger.info(res_str)

    def train_epoch(self):
        self.student.train()
        self.train_data.reset_batches(is_sequential=False)
        losses = []
        actor_losses = []
        ent_losses = []
        num_epoch = math.ceil(self.train_data.num_data / self.args['batch_size'])
        h1_list_all = []
        f1_list_all = []
        if self.use_num:
            self.question_type_list = []
        for iteration in tqdm(range(num_epoch)):
            batch = self.train_data.get_batch(iteration, self.args['batch_size'], self.args['fact_drop'])
            self.optim_student.zero_grad()
            loss, _, _, tp_list = self.student(batch, training=True)
            if self.use_num:
                self.question_type_list.extend(self.train_data.get_batch_question_types())
                num_batch = self.train_data.get_num_batch()
                loss, _, _, tp_list = self.student.forward_num(num_batch, training=True)
            h1_list, f1_list = tp_list
            h1_list_all.extend(h1_list)
            f1_list_all.extend(f1_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for name, param in self.student.named_parameters()],
                                           self.args['gradient_clip'])
            self.optim_student.step()
            losses.append(loss.item())
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all

    def save_ckpt(self, reason="h1"):
        model = self.student.model
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.ckpt".format(self.args['experiment_name'],
                                                                                   reason))
        torch.save(checkpoint, model_name)
        print("Best %s, save model as %s" %(reason, model_name))

    def load_num_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model = self.student.model.reasoning
        self.logger.info("Load param of {} from {}.".format(", ".join(list(checkpoint.keys())), filename))
        model.load_state_dict(checkpoint, strict=False)

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]
        model = self.student.model
        self.logger.info("Load param of {} from {}.".format(", ".join(list(model_state_dict.keys())), filename))
        model.load_state_dict(model_state_dict, strict=False)
