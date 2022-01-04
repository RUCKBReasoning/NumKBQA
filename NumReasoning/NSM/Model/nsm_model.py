import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from NSM.Model.base_model import BaseModel
from NSM.Modules.Instruction.seq_instruction import LSTMInstruction
from NSM.Modules.Reasoning.gnn_reasoning import GNNReasoning
from NSM.Modules.Reasoning.num_reasoning import NumReasoning
from NSM.Modules.Reasoning.num_nsm_reasoning_new import NumNSMReasoning

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class GNNModel(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(GNNModel, self).__init__(args, num_entity, num_relation, num_word)
        self.embedding_def()
        self.share_module_def()
        self.use_num = args['use_num']
        self.use_nsm_num = args['use_nsm_num']
        self.private_module_def(args, num_entity, num_relation)
        self.loss_type = "kl"
        self.model_name = args['model_name'].lower()
        self.lambda_label = args['lambda_label']
        self.filter_label = args['filter_label']
        self.to(self.device)

    def private_module_def(self, args, num_entity, num_relation):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        if self.use_num:
            if self.use_nsm_num:
                self.reasoning = NumNSMReasoning(args, num_entity, num_relation)
            else:
                self.reasoning = NumReasoning(args, num_entity, num_relation)
        else:
            self.reasoning = GNNReasoning(args, num_entity, num_relation)
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=self.rel_features)

    def one_step(self, num_step):
        # relational_ins, attn_weight = self.instruction.get_instruction(self.relational_ins, query_mask, step=num_step)
        relational_ins = self.instruction_list[num_step]
        # attn_weight = self.attn_list[num_step]
        # self.relational_ins = relational_ins
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def train_batch(self, batch, middle_dist, label_valid=None):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        # loss, extras = self.calc_loss_basic(answer_dist)
        pred_dist = self.dist_history[-1]
        # main_loss = self.get_loss_new(pred_dist, answer_dist)
        # tp_loss = self.get_loss_kl(pred_dist, answer_dist)
        # (batch_size, max_local_entity)
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # main_loss = torch.sum(tp_loss * case_valid) / pred_dist.size(0)
        main_loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        distill_loss = None
        for i in range(self.num_step - 1):
            curr_dist = self.dist_history[i + 1]
            # teacher_dist = middle_dist[i].detach()
            teacher_dist = middle_dist[i].squeeze(1).detach()
            if self.filter_label:
                assert not (label_valid is None)
                tp_label_loss = self.calc_loss_label(curr_dist=curr_dist,
                                                     teacher_dist=teacher_dist,
                                                     label_valid=label_valid)
            else:
                # tp_label_loss = self.get_loss_new(curr_dist, teacher_dist)
                tp_label_loss = self.calc_loss_label(curr_dist=curr_dist,
                                                     teacher_dist=teacher_dist,
                                                     label_valid=case_valid)
            if distill_loss is None:
                distill_loss = tp_label_loss
            else:
                distill_loss += tp_label_loss
        # pred = torch.max(pred_dist, dim=1)[1]
        extras = [main_loss.item(), distill_loss.item()]
        # tp_list = [h1.tolist(), f1.tolist()]
        loss = main_loss + distill_loss * self.lambda_label
        h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
        tp_list = [h1.tolist(), f1.tolist()]
        return loss, extras, pred_dist, tp_list

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        pred_dist = self.dist_history[-1]
        # loss, extras = self.calc_loss_basic(answer_dist)
        # tp_loss = self.get_loss_kl(pred_dist, answer_dist)
        # (batch_size, max_local_entity)
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = torch.sum(tp_loss * case_valid) / pred_dist.size(0)
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pred = torch.max(pred_dist, dim=1)[1]
        self.answer_dist = answer_dist
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list

    def forward_num(self, num_batch, training=False):
        assert self.use_num
        pred_dist_num = self.reasoning.num_layer(num_batch)
        answer_dist = self.answer_dist

        question_tokenized, attention_mask, local_entity_num, kb_fact_rel_num, q2e_adj_mat_num, \
        number_indices, number_order, repeat_entity_map, \
        repeat_entity_index, all_number_mask, num_entity_mask, \
        e2f_batch_num, e2f_f_num, e2f_e_num, e2f_val_num, f2e_batch_num, f2e_e_num, f2e_f_num, f2e_val_num, \
        judge_question, local_number_indices = num_batch
        judge_question = torch.FloatTensor(judge_question).to(self.device)

        pred_dist_fact = self.dist_history[-1]
        pred_dist = (1 - judge_question) * pred_dist_fact + judge_question * pred_dist_num

        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # main_loss = torch.sum(tp_loss * case_valid) / pred_dist.size(0)
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list