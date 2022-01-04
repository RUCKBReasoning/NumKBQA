import torch
import numpy as np
from torch.autograd import Variable, Function
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.Reasoning.base_reasoning import BaseReasoning
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


    def forward(self, d_node, d_node_mask, graph, extra_factor=None):

        d_node_len = d_node.size(1)
        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        d_node_neighbor_num = dd_graph_left.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        all_d_weight, all_q_weight = [], []
        for step in range(self.iteration_steps):
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((d_node, extra_factor), dim=-1))).squeeze(-1)

            all_d_weight.append(d_node_weight)
            self_d_node_info = self._self_node_fc(d_node)
            dd_node_info_left = self._dd_node_fc_left(d_node)

            dd_node_weight = replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dd_graph_left,
                    0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)

            agg_d_node_info = dd_node_info_left / d_node_neighbor_num.unsqueeze(-1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)

        all_d_weight = [weight.unsqueeze(1) for weight in all_d_weight]
        all_d_weight = torch.cat(all_d_weight, dim=1)

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


class NumNSMReasoning(BaseReasoning):

    def __init__(self, args, num_entity, num_relation):
        super(NumNSMReasoning, self).__init__(args, num_entity, num_relation)
        self.num_vec_func = args['num_vec_func']
        self.share_module_def()
        self.private_module_def()
        self.num_module_def(args)

        self.fact_scale = args['fact_scale']
        self.num_vec_func = args['num_vec_func']
     

    def num_module_def(self, args):
        self.node_dim = args['node_dim']
        node_dim = args['node_dim']
        from transformers import RobertaModel
        # question & rel embedding
        self.hidden_dim = 768
        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights,
                                                          cache_dir=args['cache_dir'])
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        self.hidden2ent = nn.Linear(self.hidden_dim, node_dim)

        gcn_steps = 2

        self.number_embedding = nn.Embedding(num_embeddings=self.num_entity + 2, embedding_dim=node_dim,
                                             padding_idx=self.num_entity)
        self.number_embedding.weight.requires_grad = False


        relation_embedding_file = args['relation_embedding_file']
        self.number_relation_embedding = nn.Embedding.from_pretrained(torch.from_numpy(
            np.pad(np.load(relation_embedding_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
        self.number_relation_embedding.weight.requires_grad = True
        self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
        self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
        self._iteration_steps = gcn_steps
        self._proj_ln = nn.LayerNorm(node_dim)
        self._gcn_enc = ResidualGRU(node_dim, args['num_dropout'], 2)

        self.sep_embedding = nn.Embedding(num_embeddings=1, embedding_dim=node_dim)
        self.sep_embedding.weight.requires_grad = False

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=node_dim, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=2)
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

        self.reduce_emb_linear = nn.Linear(in_features=3 * node_dim, out_features=node_dim)

        self.kb_head_linear = nn.Linear(in_features=node_dim, out_features=node_dim)
        self.kb_tail_linear = nn.Linear(in_features=node_dim, out_features=node_dim)
        self.kb_self_linear = nn.Linear(in_features=node_dim, out_features=node_dim)

        entity_dim = self.entity_dim

        self.add_module('rel_linear_num', nn.Linear(in_features=node_dim, out_features=node_dim))
        self.add_module('e2e_linear_num', nn.Linear(in_features=2 * entity_dim, out_features=entity_dim))

        # non-linear activation
        self.relu = nn.ReLU()
        if self.num_vec_func == "concat":
            self.score_func_num = nn.Linear(in_features=entity_dim * 2, out_features=1)

        if self.node_dim != self.entity_dim:
            self.number2ent = nn.Linear(node_dim, self.entity_dim)

    def load_from_pretrained(self, ckpt_file):
        checkpoint = torch.load(ckpt_file)
        model_state_dict = checkpoint["model_state_dict"]
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

    def reason_layer(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))

        possible_tail = torch.sparse.mm(self.fact2tail_mat, fact_prior)
        possible_tail = (possible_tail > VERY_SMALL_NUMBER).float().view(batch_size, max_local_entity)

        fact_val = fact_val * fact_prior
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
        self.curr_dist = current_dist
        self.curr_ins = relational_ins
        if return_score:
            return score_tp, current_dist
        return current_dist

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1, 0)
        cls_embedding = states[0]
        question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return roberta_last_hidden_states, question_embedding

    def num_layer(self, num_batch, return_score=False):
        local_entity_emb = self.local_entity_emb
        question_tokenized, attention_mask, local_entity_num, kb_fact_rel_num, q2e_adj_mat_num, \
        number_indices, number_order, repeat_entity_map, \
        repeat_entity_index, all_number_mask, num_entity_mask, \
        e2f_batch_num, e2f_f_num, e2f_e_num, e2f_val_num, f2e_batch_num, f2e_e_num, f2e_f_num, f2e_val_num, \
        judge_question, local_number_indices = num_batch

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

        local_entity_emb_num = self.number_embedding(local_entity_num)  # batch_size, max_local_entity, entity_dim
        local_fact_emb_num = self.number_relation_embedding(kb_fact_rel_num)  # batch_size, max_fact, entity_dim

        query_hidden_emb, query_node_emb = self.getQuestionEmbedding(question_tokenized, attention_mask)
        query_hidden_emb = self.hidden2ent(query_hidden_emb)  # batch_size, padding_question_length, entity_embedding
        query_node_emb = self.hidden2ent(query_node_emb).unsqueeze(dim=1)  # batch_size, 1, hidden_dim
        origin_query_node_emb = query_node_emb

       

        # num gcn
        number_indices_1 = number_indices.reshape(number_indices.shape[0] * number_indices.shape[1],
                                                  number_indices.shape[2])
        number_order_1 = number_order.reshape(number_order.shape[0] * number_order.shape[1], number_order.shape[2])
        number_mask = (number_indices_1 > -1).long()
        clamped_number_indices = replace_masked_values(number_indices_1, number_mask, self.num_entity).long()
        clamped_number_indices = clamped_number_indices.to(self.device)
        encoded_numbers = self.number_embedding(clamped_number_indices)

        # print("encoded_numbers done")

        new_graph_mask = number_order_1.unsqueeze(1).expand(number_indices_1.shape[0], number_order_1.size(-1),
                                                            -1) > number_order_1.unsqueeze(-1).expand(
            number_indices_1.shape[0], -1, number_order_1.size(-1))

        new_graph_mask = new_graph_mask.long()
        new_graph_mask = number_mask.unsqueeze(1) * number_mask.unsqueeze(-1) * new_graph_mask
        d_node, d_node_weight = self._gcn(d_node=encoded_numbers, d_node_mask=number_mask, graph=new_graph_mask)

        # print("gcn done")

        batch_num_node = d_node.unsqueeze(0)
        batch_num_node = batch_num_node.reshape(number_indices.shape[0], number_indices.shape[1], d_node.shape[-2],
                                                d_node.shape[
                                                    -1])  # batch_szie, max_length_size, num_size, entity_embedding

        # num transformer
        batch_query_hidden_emb = query_hidden_emb.unsqueeze(1)  # batch_size, padding_question_length, entity_embedding
        batch_query_hidden_emb = batch_query_hidden_emb.expand(query_hidden_emb.shape[0], number_indices.shape[1],
                                                               query_hidden_emb.shape[-2],
                                                               query_hidden_emb.shape[
                                                                   -1])  # batch_szie, max_length_size, padding_question_length, entity_embedding

        sep = torch.zeros((number_indices.shape[0], number_indices.shape[1])).type('torch.LongTensor').to(
            self.device)  # batch_szie, max_length_size,
        batch_sep_emb = self.sep_embedding(sep).unsqueeze(-2)  # batch_szie, max_length_size, 1, entity_embedding
        batch_attention_mask = attention_mask.unsqueeze(1).expand(attention_mask.shape[0], number_indices.shape[1],
                                                                  attention_mask.shape[
                                                                      1])  # batch_size,max_length_size, padding_question_length,
        batch_num_mask = (number_indices > -1).long()  # batch_szie, max_length_size, num_size
        batch_quesnum_input = torch.cat((batch_query_hidden_emb, batch_sep_emb, batch_num_node),
                                        dim=2)  # batch_szie, max_length_size,(question_size+1+number_size),embedding_dim
        batch_quesnum_mask = torch.cat((1 - batch_attention_mask, sep.unsqueeze(-1), 1 - batch_num_mask),
                                       dim=2).type('torch.ByteTensor').to(
            self.device)  # batch_szie, max_length_size, question_size+1+number_size
        quesnum_input = batch_quesnum_input.reshape(batch_quesnum_input.shape[0] * batch_quesnum_input.shape[1],
                                                    batch_quesnum_input.shape[2], batch_quesnum_input.shape[
                                                        3])  # batch_szie* max_length_size, question_size+1+number_size
        quesnum_mask = batch_quesnum_mask.reshape(batch_quesnum_mask.shape[0] * batch_quesnum_mask.shape[1],
                                                  batch_quesnum_mask.shape[
                                                      2])  # batch_szie*max_length_size, question_size+1+number_size

        quesnum_output = self.transformer_encoder(quesnum_input.permute(1, 0, 2),
                                                  src_key_padding_mask=quesnum_mask)  # (question_size+1+number_size),batch_szie*max_length_size,embedding_dim
        numq_emb = quesnum_output.permute(1, 0, 2)[:, (batch_query_hidden_emb.shape[2] + 1):,
                   :]  # batch_szie*max_length_size, num_size, embedding_size
        batch_numq_emb = numq_emb.unsqueeze(0)

        batch_numq_emb = batch_numq_emb.reshape(number_indices.shape[0], number_indices.shape[1], numq_emb.shape[-2],
                                                numq_emb.shape[
                                                    -1])  # batch_szie, max_length_size, num_size, embedding_size

        #  number embedding plugin
        batch_numq_emb_flat = batch_numq_emb.reshape(batch_numq_emb.shape[0],
                                                     batch_numq_emb.shape[1] * batch_numq_emb.shape[2],
                                                     batch_numq_emb.shape[
                                                         3])  # batch_szie, max_length_size*num_size, embedding_size

        gcn_info_vec = torch.zeros(
            (local_entity_emb_num.size(0), local_entity_emb_num.size(1) + 1, self.node_dim),
            dtype=torch.float).to(self.device)
        local_number_indices_flat = local_number_indices.reshape(number_indices.shape[0],
                                                                 number_indices.shape[1] * number_indices.shape[
                                                                     2])  # batch_szie, max_length_size*num_size,
        number_mask_flat = (local_number_indices_flat > -1).long()
        clamped_number_indices_flat = replace_masked_values(local_number_indices_flat, number_mask_flat,
                                                            gcn_info_vec.size(1) - 1)
        gcn_info_vec.scatter_(1, clamped_number_indices_flat.unsqueeze(-1).expand(-1, -1, batch_numq_emb_flat.size(-1)),
                              batch_numq_emb_flat)
        gcn_info_vec = gcn_info_vec[:, :-1, :]
        

        fact_query = query_node_emb # (batch_size, 1, dim)
        fact_gate = torch.sigmoid(self.rel_linear_num(local_fact_emb_num) * fact_query) # (batch_size, max_fact, dim) (batch_size, 1, dim)
        fact_val = sparse_bmm(entity2fact_mat_num, gcn_info_vec)# (batch_size, max_fact, max_ent) (batch_size, max_ent, dim) -> (batch_size, max_fact, dim)
        local_entity_emb_num = sparse_bmm(fact2entity_mat_num, fact_gate * fact_val)# (batch_size, max_ent, max_fact) (batch_size, max_fact, dim) -> (batch_size, max_ent, dim)
        num_info_vec = torch.zeros((batch_size, local_entity_emb.size(1) + 1, self.node_dim),
                                   dtype=torch.float, device=local_entity_emb_num.device)
        num_entity_mask_tag = (num_entity_mask != self.num_entity).type('torch.FloatTensor').to(self.device)
        clamped_number_indices = replace_masked_values(num_entity_mask, num_entity_mask_tag,
                                                       num_info_vec.size(1) - 1)
        num_info_vec.scatter_(1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, local_entity_emb_num.size(-1)),
                              local_entity_emb_num)
        num_info_vec = num_info_vec[:, :-1, :]# (batch_size, max_local_entity, dim)
        num_info_vec = num_info_vec * self.curr_dist.unsqueeze(2) # (batch_size, max_local_entity, dim) (batch_size, max_local_entity, 1) -> (batch_size, max_local_entity, dim)
        # only current distribution covered entities are considered to gain number information vectors
        if self.node_dim != self.entity_dim:
            num_info_vec = self.number2ent(num_info_vec)
        next_local_entity_emb = torch.cat((self.local_entity_emb, num_info_vec), dim=2)
        local_entity_emb = F.relu(self.e2e_linear_num(self.linear_drop(next_local_entity_emb)))

        score_tp = self.score_func(self.linear_drop(local_entity_emb)).squeeze(dim=2)
        score_tp = score_tp + (1 - self.possible_cand[-1]) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
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
