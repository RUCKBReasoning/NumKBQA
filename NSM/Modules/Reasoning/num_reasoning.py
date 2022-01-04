import torch
import numpy as np
from torch.autograd import Variable, Function
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.Reasoning.base_reasoning import BaseReasoning
# from util import sparse_bmm, replace_masked_values
# from model_util import GCN, ResidualGRU, FFNLayer
# from NumGCN import GCN, ResidualGRU, FFNLayer
from transformers import *

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        #print(input.shape)
        #print(self.dropout_func(input).shape)
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)



class GCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=3):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)
        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)

    #def forward(self, d_node, q_node, d_node_mask, q_node_mask, graph, extra_factor=None):
    def forward(self, d_node, d_node_mask, graph, extra_factor=None):

        # d_node = encoded_numbers, q_node = question_encoded_number,
        # d_node_mask = number_mask, q_node_mask = question_number_mask, graph = new_graph_mask

        d_node_len = d_node.size(1)
        #q_node_len = q_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        #print(dd_graph)
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        # print('dd_graph_left')
        # for i in range(len(dd_graph_left[1])):
        #     print(dd_graph_left[1].shape)
        #     print(dd_graph_left[1][i])
        #dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])
        #print(dd_graph_right)
        #生成判断大小矩阵

        # diagmat = torch.diagflat(torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device))
        # diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        # qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * (1 - diagmat)
        # qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        # qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])
        #
        # dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1)
        # dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        # dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])
        #
        # qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1)
        # qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        # qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])


        #d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1) + dq_graph_left.sum(-1) + dq_graph_right.sum(-1)
        #d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1)
        d_node_neighbor_num = dd_graph_left.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        # q_node_neighbor_num = qq_graph_left.sum(-1) + qq_graph_right.sum(-1) + qd_graph_left.sum(-1) + qd_graph_right.sum(-1)
        # q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        # q_node_neighbor_num = util.replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1)


        all_d_weight, all_q_weight = [], []
        for step in range(self.iteration_steps):
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)
                #q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((d_node, extra_factor), dim=-1))).squeeze(-1)
                #q_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((q_node, extra_factor), dim=-1))).squeeze(-1)

            all_d_weight.append(d_node_weight)
            #all_q_weight.append(q_node_weight)

            self_d_node_info = self._self_node_fc(d_node)
            #self_q_node_info = self._self_node_fc(q_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)
            #qd_node_info_left = self._qd_node_fc_left(d_node)
            #qq_node_info_left = self._qq_node_fc_left(q_node)
            #dq_node_info_left = self._dq_node_fc_left(q_node)
            # print('dd_node_info_left')
            # print(dd_node_info_left.shape)
            # print(dd_node_info_left[1])

            dd_node_weight = replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dd_graph_left,
                    0)

            #dd_node_weight = F.softmax(dd_node_weight, -1)
            # print('dd_node_weight')
            # print(dd_node_weight.shape)
            # print(dd_node_weight[1])

            # qd_node_weight = util.replace_masked_values(
            #         d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
            #         qd_graph_left,
            #         0)
            #
            # qq_node_weight = util.replace_masked_values(
            #         q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
            #         qq_graph_left,
            #         0)
            #
            # dq_node_weight = util.replace_masked_values(
            #         q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
            #         dq_graph_left,
            #         0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)
            #qd_node_info_left = torch.matmul(qd_node_weight, qd_node_info_left)
            #qq_node_info_left = torch.matmul(qq_node_weight, qq_node_info_left)
            #dq_node_info_left = torch.matmul(dq_node_weight, dq_node_info_left)


            # dd_node_info_right = self._dd_node_fc_right(d_node)
            # #qd_node_info_right = self._qd_node_fc_right(d_node)
            # #qq_node_info_right = self._qq_node_fc_right(q_node)
            # #dq_node_info_right = self._dq_node_fc_right(q_node)
            #
            # dd_node_weight = util.replace_masked_values(
            #         d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
            #         dd_graph_right,
            #         0)

            # qd_node_weight = util.replace_masked_values(
            #         d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
            #         qd_graph_right,
            #         0)
            #
            # qq_node_weight = util.replace_masked_values(
            #         q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
            #         qq_graph_right,
            #         0)
            #
            # dq_node_weight = util.replace_masked_values(
            #         q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
            #         dq_graph_right,
            #         0)

            # dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)
            #qd_node_info_right = torch.matmul(qd_node_weight, qd_node_info_right)
            #qq_node_info_right = torch.matmul(qq_node_weight, qq_node_info_right)
            #dq_node_info_right = torch.matmul(dq_node_weight, dq_node_info_right)

            #agg_d_node_info = (dd_node_info_left + dd_node_in fo_right) / d_node_neighbor_num.unsqueeze(-1)


            agg_d_node_info = dd_node_info_left / d_node_neighbor_num.unsqueeze(-1)
            # agg_d_node_info = dd_node_info_left


            #agg_d_node_info = (dd_node_info_left + dd_node_info_right + dq_node_info_left + dq_node_info_right) / d_node_neighbor_num.unsqueeze(-1)
            #agg_q_node_info = (qq_node_info_left + qq_node_info_right + qd_node_info_left + qd_node_info_right) / q_node_neighbor_num.unsqueeze(-1)

            #d_node = F.relu(self_d_node_info + agg_d_node_info +  d_node)
            d_node = F.relu(self_d_node_info + agg_d_node_info)
            #m = nn.LeakyReLU(0.1)
            #d_node = m(self_d_node_info + agg_d_node_info)
            #d_node = F.relu(self_d_node_info + agg_d_node_info) + d_node
            #d_node = self_d_node_info + agg_d_node_info
            #q_node = F.relu(self_q_node_info + agg_q_node_info)


        all_d_weight = [weight.unsqueeze(1) for weight in all_d_weight]
        #all_q_weight = [weight.unsqueeze(1) for weight in all_q_weight]

        all_d_weight = torch.cat(all_d_weight, dim=1)
        #all_q_weight = torch.cat(all_q_weight, dim=1)

        return d_node, all_d_weight


def use_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var
    return var


class LeftMMFixed(Function):  # torch.autograd.Function
    """
    Implementation of matrix multiplication of a Sparse Variable with a Dense Variable, returning a Dense one.
    This is added because there's no autograd for sparse yet. No gradient computed on the sparse weights.
    """

    #     def __init__(self):
    #         super(LeftMMFixed, self).__init__()
    #         self.sparse_weights = None

    #     @staticmethod
    #     def forward(self, sparse_weights, x):
    #         if self.sparse_weights is None:
    #             self.sparse_weights = sparse_weights
    #         return torch.mm(self.sparse_weights, x)
    #     @staticmethod
    #     def backward(self, grad_output):
    #         sparse_weights = self.sparse_weights
    #         return None, torch.mm(sparse_weights.t(), grad_output)

    @staticmethod
    def forward(ctx, sparse_weights, x):
        ctx.sparse_weights = sparse_weights
        return torch.mm(sparse_weights, x)

    @staticmethod
    def backward(ctx, grad_output):
        # sparse_weights = self.sparse_weights
        return None, torch.mm(ctx.sparse_weights.t(), grad_output)


def sparse_bmm(X, Y):
    """Batch multiply X and Y where X is sparse, Y is dense.
    Args:
        X: Sparse tensor of size BxMxN. Consists of two tensors,
            I:3xZ indices, and V:1xZ values.
        Y: Dense tensor of size BxNxK.
    Returns:
        batched-matmul(X, Y): BxMxK
    """
    I = X._indices()
    V = X._values()
    B, M, N = X.size()
    _, _, K = Y.size()
    Z = I.size()[1]
    lookup = Y[I[0, :], I[2, :], :]
    X_I = torch.stack((I[0, :] * M + I[1, :], use_cuda(torch.arange(Z).type(torch.LongTensor))), 0)
    S = use_cuda(Variable(torch.cuda.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))
    prod_op = LeftMMFixed()
    prod = prod_op.apply(S, lookup)
    #print(prod)
    return prod.view(B, M, K)


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).bool(), replace_with)


class NumReasoning(BaseReasoning):

    def __init__(self, args, num_entity, num_relation):
        super(NumReasoning, self).__init__(args, num_entity, num_relation)
        self.num_vec_func = args['num_vec_func']
        self.share_module_def()
        self.private_module_def()
        self.num_module_def(args)

        self.fact_scale = args['fact_scale']
        self.num_vec_func = args['num_vec_func']
        # self.relation2id = relation2id
        # self.id2relation = {i: relation for relation, i in relation2id.items()}
        # self.consrel2id = consrel2id
        # self.id2consrel = {i: relation for relation, i in consrel2id.items()}

    def num_module_def(self, args):
        entity_dim = self.entity_dim
        from transformers import RobertaModel
        # question & rel embedding
        self.hidden_dim = 768
        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights,
                                                          cache_dir=args['cache_dir'])
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        self.hidden2ent = nn.Linear(self.hidden_dim, self.entity_dim)

        gcn_steps = 2

        # entity_embedding_file = args['entity_embedding_file']

        self.number_embedding = nn.Embedding(num_embeddings=self.num_entity + 1, embedding_dim=self.entity_dim,
                                             padding_idx=self.num_entity)

        # self.number_embedding = nn.Embedding.from_pretrained(
        #     torch.from_numpy(np.load(entity_embedding_file)).type('torch.FloatTensor'), freeze=True)
        self.number_embedding.weight.requires_grad = False

        # self.entity_embedding.weight = nn.Parameter(
        #     torch.from_numpy(np.pad(np.load(relation_embedding_file), ((0, 1), (0, 0)), 'constant')).type(
        #         'torch.FloatTensor'))

        relation_embedding_file = args['relation_embedding_file']
        self.number_relation_embedding = nn.Embedding.from_pretrained(torch.from_numpy(
            np.pad(np.load(relation_embedding_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
        self.number_relation_embedding.weight.requires_grad = True
        # self.number_relation_embedding = nn.Embedding(num_embeddings=6128 + 1, embedding_dim=self.entity_dim,
        #                                               padding_idx=6128)

        node_dim = entity_dim  # 2:for webqsp 100d / 3: for cwq 300d
        self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
        self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
        self._iteration_steps = gcn_steps
        self._proj_ln = nn.LayerNorm(node_dim)
        # self._proj_ln0 = nn.LayerNorm(node_dim)
        # self._proj_ln1 = nn.LayerNorm(node_dim)
        # self._proj_ln3 = nn.LayerNorm(node_dim)
        self._gcn_enc = ResidualGRU(node_dim, args['num_dropout'], 2)

        self.sep_embedding = nn.Embedding(num_embeddings=1, embedding_dim=entity_dim)
        self.sep_embedding.weight.requires_grad = False

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=entity_dim, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=2)
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

        self.reduce_emb_linear = nn.Linear(in_features=3 * entity_dim, out_features=entity_dim)

        self.kb_head_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.kb_tail_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.kb_self_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)

        # non-linear activation
        self.relu = nn.ReLU()
        if self.num_vec_func == "concat":
            self.score_func_num = nn.Linear(in_features=entity_dim * 2, out_features=1)

    def load_from_pretrained(self, ckpt_file):
        checkpoint = torch.load(ckpt_file)
        model_state_dict = checkpoint["model_state_dict"]
        # model = self.student
        self.logger.info("Number reasoning, Load param of {} from {}.".format(", ".join(list(model_state_dict.keys())),
                                                                              filename))
        self.load_state_dict(model_state_dict, strict=False)

    def private_module_def(self):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2 * entity_dim, out_features=entity_dim))
            # self.add_module('score_func' + str(i), nn.Linear(in_features=entity_dim, out_features=1))

    def reason_layer(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))

        possible_tail = torch.sparse.mm(self.fact2tail_mat, fact_prior)
        # (batch_size *max_local_entity, num_fact) (num_fact, 1)
        possible_tail = (possible_tail > VERY_SMALL_NUMBER).float().view(batch_size, max_local_entity)

        fact_val = fact_val * fact_prior
        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        return neighbor_rep, possible_tail

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        # score_func = getattr(self, 'score_func' + str(step))
        score_func = self.score_func
        relational_ins = relational_ins.squeeze(1)
        neighbor_rep, possible_tail = self.reason_layer(current_dist, relational_ins, rel_linear)
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_rep), dim=2)
        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))

        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        if self.reason_kb:
            answer_mask = self.local_entity_mask * possible_tail
        else:
            answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        return current_dist

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        # print(roberta_last_hidden_states.shape)
        states = roberta_last_hidden_states.transpose(1, 0)
        cls_embedding = states[0]
        #question_embedding = cls_embedding
        question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return roberta_last_hidden_states, question_embedding

    def num_layer(self, num_batch, return_score=False):
        local_entity_emb = self.local_entity_emb
        question_tokenized, attention_mask, local_entity_num, kb_fact_rel_num, q2e_adj_mat_num, \
        number_indices, number_order, repeat_entity_map, \
        repeat_entity_index, all_number_mask, num_entity_mask, \
        e2f_batch_num, e2f_f_num, e2f_e_num, e2f_val_num, f2e_batch_num, f2e_e_num, f2e_f_num, f2e_val_num, \
        judge_question, local_number_indices = num_batch

        # data process
        # question_tokenized = torch.LongTensor(question_tokenized).to(self.device)
        # attention_mask = torch.FloatTensor(attention_mask).to(self.device)
        num_entity_mask = torch.LongTensor(num_entity_mask).to(self.device)
        local_entity_num = torch.LongTensor(local_entity_num).to(self.device)
        kb_fact_rel_num = torch.LongTensor(kb_fact_rel_num).to(self.device)
        q2e_adj_mat_num = torch.LongTensor(q2e_adj_mat_num).to(self.device)

        question_tokenized = question_tokenized.type('torch.LongTensor').to(self.device)
        attention_mask = attention_mask.type('torch.LongTensor').to(self.device)
        local_number_indices = torch.LongTensor(local_number_indices).to(self.device)
        number_indices = torch.LongTensor(number_indices).to(self.device)
        number_order = torch.LongTensor(number_order).to(self.device)
        all_number_mask = torch.LongTensor(all_number_mask).to(self.device)


        # print(local_fact_emb_num.shape)

        e2f_batch_num = np.array(sum(e2f_batch_num, []))
        e2f_f_num = np.array(sum(e2f_f_num, []))
        e2f_e_num = np.array(sum(e2f_e_num, []))
        e2f_val_num = np.array(sum(e2f_val_num, []))
        f2e_batch_num = np.array(sum(f2e_batch_num, []))
        f2e_e_num = np.array(sum(f2e_e_num, []))
        f2e_f_num = np.array(sum(f2e_f_num, []))
        f2e_val_num = np.array(sum(f2e_val_num, []))

        batch_size, max_num_local_entity = local_entity_num.shape
        _, max_num_fact = kb_fact_rel_num.shape

        # print(e2f_batch_num.shape)
        #
        # print(e2f_f_num.shape)
        # print(e2f_e_num.shape)
        #
        # print(e2f_batch_num)

        entity2fact_index_num = torch.LongTensor([e2f_batch_num, e2f_f_num, e2f_e_num])
        entity2fact_val_num = torch.FloatTensor(e2f_val_num)
        entity2fact_mat_num = torch.sparse.FloatTensor(entity2fact_index_num, entity2fact_val_num,
                                                       torch.Size([batch_size, max_num_fact, max_num_local_entity])).to(
            self.device)  # batch_size, max_fact, max_local_entity

        fact2entity_index_num = torch.LongTensor([f2e_batch_num, f2e_e_num, f2e_f_num])
        fact2entity_val_num = torch.FloatTensor(f2e_val_num)
        fact2entity_mat_num = torch.sparse.FloatTensor(fact2entity_index_num, fact2entity_val_num,
                                                       torch.Size([batch_size, max_num_local_entity, max_num_fact])).to(
            self.device)

        pagerank_f_num = Variable(q2e_adj_mat_num.type('torch.FloatTensor'), requires_grad=False).squeeze(dim=2).to(
            self.device)

        # print("Num_load_data:{:.4}".format( time.time() - st))

        # process embedding
        local_entity_emb_num = self.number_embedding(local_entity_num)  # batch_size, max_local_entity, entity_dim
        local_fact_emb_num = self.number_relation_embedding(kb_fact_rel_num)  # batch_size, max_fact, entity_dim

        query_hidden_emb, query_node_emb = self.getQuestionEmbedding(question_tokenized, attention_mask)
        query_hidden_emb = self.hidden2ent(query_hidden_emb)  # batch_size, padding_question_length, entity_embedding
        query_node_emb = self.hidden2ent(query_node_emb).unsqueeze(dim=1)  # batch_size, 1, hidden_dim
        origin_query_node_emb = query_node_emb

        # calculate relation & entity weights
        div = float(np.sqrt(self.entity_dim))
        fact2query_sim_num = torch.bmm(query_hidden_emb, local_fact_emb_num.transpose(1, 2)) / div
        # batch_size, max_query_word, max_fact
        fact2query_sim_num = self.softmax_d1(fact2query_sim_num +
                                             (1 - attention_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER)
        # batch_size, max_query_word, max_fact
        fact2query_att_num = torch.sum(fact2query_sim_num.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2), dim=1)
        # batch_size, max_fact, entity_dim
        W_num = torch.sum(fact2query_att_num * local_fact_emb_num, dim=2) / div
        # batch_size, max_fact
        W_max_num = torch.max(W_num, dim=1, keepdim=True)[0]
        # batch_size, 1
        W_tilde_num = torch.exp(W_num - W_max_num)
        # batch_size, max_fact
        e2f_softmax_num = sparse_bmm(entity2fact_mat_num.transpose(1, 2), W_tilde_num.unsqueeze(dim=2)).squeeze(dim=2)
        # batch_size, max_local_entity
        e2f_softmax_num = torch.clamp(e2f_softmax_num, min=VERY_SMALL_NUMBER)

        # print("Query_relation_weight:{:.4}".format(time.time()  - st))

        # num gcn
        # num_mask  = np.zeros((local_entity_emb_num.shape[0], local_entity_emb_num.shape[1]), dtype=float)
        number_indices_1 = number_indices.reshape(number_indices.shape[0] * number_indices.shape[1],
                                                  number_indices.shape[2])
        number_order_1 = number_order.reshape(number_order.shape[0] * number_order.shape[1], number_order.shape[2])
        number_mask = (number_indices_1 > -1).long()
        clamped_number_indices = replace_masked_values(number_indices_1, number_mask, self.num_entity).long()
        clamped_number_indices = clamped_number_indices.to(self.device)
        encoded_numbers = self.number_embedding(clamped_number_indices)
        # number_order_1 = torch.LongTensor(number_order_1).to(self.device)

        new_graph_mask = number_order_1.unsqueeze(1).expand(number_indices_1.shape[0], number_order_1.size(-1),
                                                            -1) > number_order_1.unsqueeze(-1).expand(
            number_indices_1.shape[0], -1, number_order_1.size(-1))

        new_graph_mask = new_graph_mask.long()
        new_graph_mask = number_mask.unsqueeze(1) * number_mask.unsqueeze(-1) * new_graph_mask
        d_node, d_node_weight = self._gcn(d_node=encoded_numbers, d_node_mask=number_mask, graph=new_graph_mask)

        batch_num_node = d_node.unsqueeze(0)
        batch_num_node = batch_num_node.reshape(number_indices.shape[0], number_indices.shape[1], d_node.shape[-2],
                                                d_node.shape[
                                                    -1])  # batch_szie, max_length_size, num_size, entity_embedding
        # print(batch_num_node.shape)

        # print("Num_gcn:{:.4}".format(time.time() - st))

        # num transformer
        batch_query_hidden_emb = query_hidden_emb.unsqueeze(1)  # batch_size, padding_question_length, entity_embedding
        batch_query_hidden_emb = batch_query_hidden_emb.expand(query_hidden_emb.shape[0], number_indices.shape[1],
                                                               query_hidden_emb.shape[-2],
                                                               query_hidden_emb.shape[
                                                                   -1])  # batch_szie, max_length_size, padding_question_length, entity_embedding
        # print(batch_query_hidden_emb.shape)
        sep = torch.zeros((number_indices.shape[0], number_indices.shape[1])).type('torch.LongTensor').to(
            self.device)  # batch_szie, max_length_size,
        # print(sep.shape)
        batch_sep_emb = self.sep_embedding(sep).unsqueeze(-2)  # batch_szie, max_length_size, 1, entity_embedding
        # print(batch_sep_emb.shape)

        batch_attention_mask = attention_mask.unsqueeze(1).expand(attention_mask.shape[0], number_indices.shape[1],
                                                                  attention_mask.shape[
                                                                      1])  # batch_size,max_length_size, padding_question_length,

        # print(batch_attention_mask.shape)

        batch_num_mask = (number_indices > -1).long()  # batch_szie, max_length_size, num_size

        # print(batch_num_mask.shape)

        batch_quesnum_input = torch.cat((batch_query_hidden_emb, batch_sep_emb, batch_num_node),
                                        dim=2)  # batch_szie, max_length_size,(question_size+1+number_size),embedding_dim
        # print(batch_quesnum_input.shape)
        batch_quesnum_mask = torch.cat((1 - batch_attention_mask, sep.unsqueeze(-1), 1 - batch_num_mask),
                                       dim=2).type('torch.ByteTensor').to(
            self.device)  # batch_szie, max_length_size, question_size+1+number_size
        # print(batch_quesnum_mask.shape)
        quesnum_input = batch_quesnum_input.reshape(batch_quesnum_input.shape[0] * batch_quesnum_input.shape[1],
                                                    batch_quesnum_input.shape[2], batch_quesnum_input.shape[
                                                        3])  # batch_szie* max_length_size, question_size+1+number_size

        # print(quesnum_input.shape)
        quesnum_mask = batch_quesnum_mask.reshape(batch_quesnum_mask.shape[0] * batch_quesnum_mask.shape[1],
                                                  batch_quesnum_mask.shape[
                                                      2])  # batch_szie*max_length_size, question_size+1+number_size
        # print(quesnum_mask.shape)
        quesnum_output = self.transformer_encoder(quesnum_input.permute(1, 0, 2),
                                                  src_key_padding_mask=quesnum_mask)  # (question_size+1+number_size),batch_szie*max_length_size,embedding_dim
        # print(quesnum_output.shape)
        numq_emb = quesnum_output.permute(1, 0, 2)[:, (batch_query_hidden_emb.shape[2] + 1):,
                   :]  # batch_szie*max_length_size, num_size, embedding_size
        # print(numq_emb.shape)
        batch_numq_emb = numq_emb.unsqueeze(0)

        batch_numq_emb = batch_numq_emb.reshape(number_indices.shape[0], number_indices.shape[1], numq_emb.shape[-2],
                                                numq_emb.shape[
                                                    -1])  # batch_szie, max_length_size, num_size, embedding_size

        # print("Num_transformers:{:.4}".format(time.time() - st))

        #  number embedding plugin
        batch_numq_emb_flat = batch_numq_emb.reshape(batch_numq_emb.shape[0],
                                                     batch_numq_emb.shape[1] * batch_numq_emb.shape[2],
                                                     batch_numq_emb.shape[
                                                         3])  # batch_szie, max_length_size*num_size, embedding_size

        # print(batch_numq_emb.shape)

        gcn_info_vec = torch.zeros(
            (local_entity_emb_num.size(0), local_entity_emb_num.size(1) + 1, local_entity_emb_num.size(-1)),
            dtype=torch.float).to(self.device)

        # print(gcn_info_vec.shape)

        local_number_indices_flat = local_number_indices.reshape(number_indices.shape[0],
                                                                 number_indices.shape[1] * number_indices.shape[
                                                                     2])  # batch_szie, max_length_size*num_size,
        # print(local_number_indices_flat.shape)
        number_mask_flat = (local_number_indices_flat > -1).long()

        # print(number_mask_flat.shape)
        # print(number_mask_flat)

        clamped_number_indices_flat = replace_masked_values(local_number_indices_flat, number_mask_flat,
                                                            gcn_info_vec.size(1) - 1)
        # print(clamped_number_indices_flat.unsqueeze(-1).expand(-1, -1, batch_numq_emb_flat.size(-1)))
        gcn_info_vec.scatter_(1, clamped_number_indices_flat.unsqueeze(-1).expand(-1, -1, batch_numq_emb_flat.size(-1)),
                              batch_numq_emb_flat)
        # print(gcn_info_vec.shape)
        # print(gcn_info_vec)
        gcn_info_vec = gcn_info_vec[:, :-1, :]
        # print(gcn_info_vec.shape)

        local_entity_emb_num = local_entity_emb_num * (1 - all_number_mask).unsqueeze(
            dim=2) + gcn_info_vec  # print(1,2,local_entity_emb.shape)

        # print(local_entity_emb_num.shape)
        # print(entity2fact_mat_num.shape)
        e2f_emb_num = self.relu(
            self.kb_self_linear(local_fact_emb_num) + sparse_bmm(entity2fact_mat_num, self.kb_head_linear(
                self.linear_drop(local_entity_emb_num))))  # batch_size, max_fact, entity_dim

        e2f_softmax_normalized = W_tilde_num.unsqueeze(dim=2) * sparse_bmm(entity2fact_mat_num,
                                                                           (pagerank_f_num / e2f_softmax_num).unsqueeze(
                                                                               dim=2))
        # * all_repeat_fact_mask.unsqueeze(-1) # batch_size, max_fact, 1

        e2f_emb_num = e2f_emb_num * e2f_softmax_normalized  # batch_size, max_fact, entity_dim
        f2e_emb_num = self.relu(sparse_bmm(fact2entity_mat_num, self.kb_tail_linear(
            self.linear_drop(e2f_emb_num))))
        local_entity_emb_num = self.fact_scale * f2e_emb_num  # batch_size, max_local_entity, entity_dim

        num_info_vec = torch.zeros((batch_size, local_entity_emb.size(1) + 1, local_entity_emb.size(-1)),
                                   dtype=torch.float, device=local_entity_emb_num.device)

        num_entity_mask_tag = (num_entity_mask != self.num_entity).type('torch.FloatTensor').to(self.device)
        # print(num_entity_mask_tag)
        # print(num_entity_mask)
        clamped_number_indices = replace_masked_values(num_entity_mask, num_entity_mask_tag,
                                                       num_info_vec.size(1) - 1)
        # print(clamped_number_indices)

        num_info_vec.scatter_(1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, local_entity_emb_num.size(-1)),
                              local_entity_emb_num)
        num_info_vec = num_info_vec[:, :-1, :]

        # score_tp = self.score_func(self.linear_drop(local_entity_emb + num_info_vec)).squeeze(dim=2)
        if self.num_vec_func == "add":
            score_tp = self.score_func(self.linear_drop(local_entity_emb + num_info_vec)).squeeze(dim=2)
        elif self.num_vec_func == "concat":
            concat_vec = torch.cat((local_entity_emb, num_info_vec), dim=-1)
            score_tp = self.score_func_num(self.linear_drop(concat_vec)).squeeze(dim=2)
        else:
            raise NotImplementedError("Unknown number usage : {}".format(self.num_vec_func))

        score_tp = score_tp + (1 - self.possible_cand[-1]) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        # judge_question * current_dist + (1 - judge_question) * self.
        if return_score:
            return score_tp, current_dist
        return current_dist

    def forward_all(self, curr_dist, instruction_list):
        dist_history = [curr_dist]
        score_list = []
        for i in range(self.num_step):
            score_tp, curr_dist = self.forward(curr_dist, instruction_list[i], step=i, return_score=True)
            if i == self.num_step - 1:
                score_tp, curr_dist = self.num_layer(num_batch, local_entity_emb=self.local_entity_emb)
            score_list.append(score_tp)
            dist_history.append(curr_dist)
        return dist_history, score_list

    # def __repr__(self):
    #     return "GNN based reasoning"
