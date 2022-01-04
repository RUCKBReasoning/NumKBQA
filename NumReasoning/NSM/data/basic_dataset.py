import json
import numpy as np
import re
import os
from tqdm import tqdm
import torch
from NSM.data.read_tree import read_tree
from collections import Counter
from collections import defaultdict

num_entity_threshold = 110
num_entity_threshold_all = 110
num_fact_threshold = 110


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id


class BasicDataLoader(object):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        self._parse_args(config, word2id, relation2id, entity2id)
        self._load_file(config, data_type)
        self._load_data()
        if config['use_num']:
            self._load_expan_relation(config)
            self._load_num()

    def _load_expan_relation(self, config):
        relation_file = config['data_folder'] + "relations_expanded.txt"
        if not os.path.exists(relation_file):
            relation_file = config['data_folder'] + "relations.txt"
        self.expand_relation2id = load_dict(relation_file)
        self.expand_id2relation = {i: relation for relation, i in self.expand_relation2id.items()}

    def _load_file(self, config, data_type="train"):
        data_file = config['data_folder'] + data_type + "_simple.json"
        num_file = config['data_folder'] + data_type + ".num_new"
        print('loading data from', data_file)
        self.data = []
        self.number_data = []
        skip_index = set()
        index = 0
        with open(data_file) as f_in:
            for line in tqdm(f_in):
                index += 1
                line = json.loads(line)
                try:
                    if len(line['entities']) == 0:
                        skip_index.add(index)
                        continue
                except:
                    print(line['entities'])
                    skip_index.add(index)
                    continue
                self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
        print("skip", skip_index)
        print('max_facts: ', self.max_facts)
        self.num_entity_threshold = 110
        self.num_entity_threshold_all = 110
        self.num_fact_threshold = 110
        self.cons_question = set()
        index = 0
        self.question_type_list = []
        with open(num_file) as f_in:
            for line in f_in:
                index += 1
                if index in skip_index:
                    continue
                line = json.loads(line)
                if "question_type" in line:
                    self.question_type_list.append(line["question_type"])
                else:
                    self.question_type_list.append("unknown")
                if "question_type" in line and line["question_type"] in ["comparative", "superlative"]:
                    self.cons_question.add(line["id"])
                self.number_data.append(line)
            assert len(self.question_type_list) == len(self.data)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

    def _load_data(self):
        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()

        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity

        self.question_id = []
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.q_adj_mats = np.empty(self.num_data, dtype=object)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.seed_list = np.empty(self.num_data, dtype=object)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_lists = np.empty(self.num_data, dtype=object)
        if self.use_num:
            from transformers import RobertaTokenizer
            self.padding_question_length = 128
            self.question_token = torch.zeros(self.num_data, self.padding_question_length, dtype=int)
            self.question_attention_mask = torch.zeros(self.num_data, self.padding_question_length)
            self.tokenizer_class = RobertaTokenizer
            self.pretrained_weights = 'roberta-base'
            self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights,
                                                                  cache_dir=self.cache_dir)

        print('preparing question ...')
        self._prepare_question()
        print('preparing data ...')
        self._prepare_data()

    def _parse_args(self, config, word2id, relation2id, entity2id):
        self.use_inverse_relation = config['use_inverse_relation']
        self.data_folder = config['data_folder']
        self.use_self_loop = config['use_self_loop']
        self.num_step = config['num_step']
        self.max_local_entity = 0
        self.max_relevant_doc = 0
        self.max_facts = 0
        self.use_num = config['use_num']
        self.cache_dir = config['cache_dir']

        print('building word index ...')
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2relation = {i: relation for relation, i in relation2id.items()}
        self.id2entity = {i: entity for entity, i in entity2id.items()}
        self.q_type = config['q_type']

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        print("Entity: {}, Relation in KB: {}, Relation in use: {} ".format(len(entity2id),
                                                                            len(self.relation2id),
                                                                            self.num_kb_relation))

    @staticmethod
    def tokenize_sent(question_text):
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        for w_idx, w in enumerate(question_text.split(' ')):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w == '':
                continue
            words += [w]
        return words

    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def get_num_batch(self):
        return self._build_number_mat(self.sample_ids)

    def get_quest(self, training=False):
        q_list = []
        sample_ids = self.sample_ids
        for sample_id in sample_ids:
            tp_str = self.decode_text(self.query_texts[sample_id, :])
            q_list.append(tp_str)
        return q_list

    def decode_text(self, np_array_x):
        id2word = self.id2word
        tp_str = ""
        for i in range(self.max_query_word):
            if np_array_x[i] in id2word:
                tp_str += id2word[np_array_x[i]] + " "
        return tp_str

    def _prepare_question(self):
        max_count = 0
        tokens_list = []
        for sample in tqdm(self.data):
            word_list = self.tokenize_sent(question_text=sample['question'])
            tokens_list.append(word_list)
            max_count = max(max_count, len(word_list))
        self.max_query_word = max_count
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        next_id = 0
        for tokens in tokens_list:
            for j, word in enumerate(tokens):
                if word in self.word2id:
                    self.query_texts[next_id, j] = self.word2id[word]
                else:
                    self.query_texts[next_id, j] = len(self.word2id)
            next_id += 1

    def _prepare_dep(self):
        max_count = 0
        for line in self.dep:
            word_list = line["dep"]
            max_count = max(max_count, len(word_list))
        self.max_query_word = max_count
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        next_id = 0
        self.node2layer = []
        for sample in tqdm(self.dep):
            tp_dep = sample["dep"]
            node_layer, parents, relations = read_tree(tp_dep)
            self.node2layer.append(node_layer)
            tokens = [item[0] for item in tp_dep]
            for j, word in enumerate(tokens):
                if word in self.word2id:
                    self.query_texts[next_id, j] = self.word2id[word]
                else:
                    self.query_texts[next_id, j] = len(self.word2id)
            next_id += 1

    def tokenize_question_roberta(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 128)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        num_query_entity = {}
        for sample in tqdm(self.data):
            self.question_id.append(sample["id"])

            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            if len(g2l) == 0:
                print(next_id)
                continue

            if self.use_num:
                question = sample['question'] + ' NE'
                question_text = question.strip()
                question_tokenized, attention_mask = self.tokenize_question_roberta(question_text)
                try:
                    self.question_token[next_id] = question_tokenized
                    self.question_attention_mask[next_id] = attention_mask
                except:
                    print(question_text)
                    print(question_tokenized)
                    print(question_tokenized.size())

            tp_set = set()
            seed_list = []
            for j, entity in enumerate(sample['entities']):
                global_entity = entity  # self.entity2id[entity['text']]
                if global_entity not in g2l:
                    continue
                local_ent = g2l[global_entity]
                self.query_entities[next_id, local_ent] = 1.0
                seed_list.append(local_ent)
                tp_set.add(local_ent)
            self.seed_list[next_id] = seed_list
            num_query_entity[next_id] = len(tp_set)
            for global_entity, local_entity in g2l.items():
                if local_entity not in tp_set:  # skip entities in question
                    self.candidate_entities[next_id, local_entity] = global_entity

            # relations in local KB
            head_list = []
            rel_list = []
            tail_list = []
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                head = g2l[sbj]
                rel = int(rel)
                tail = g2l[obj]
                head_list.append(head)
                rel_list.append(rel)
                tail_list.append(tail)
                if self.use_inverse_relation:
                    head_list.append(tail)
                    rel_list.append(rel + len(self.relation2id))
                    tail_list.append(head)
            if len(tp_set) > 0:
                for local_ent in tp_set:
                    self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
            else:
                for index in range(len(g2l)):
                    self.seed_distribution[next_id, index] = 1.0 / len(g2l)
            try:
                assert np.sum(self.seed_distribution[next_id]) > 0.0
            except:
                print(next_id, len(tp_set))
                exit(-1)

            answer_list = []
            for answer in sample['answers']:
                keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                answer_ent = self.entity2id[answer[keyword]]
                answer_list.append(answer_ent)
                if answer_ent in g2l:
                    self.answer_dists[next_id, g2l[answer_ent]] = 1.0
            self.answer_lists[next_id] = answer_list
            self.kb_adj_mats[next_id] = (np.array(head_list, dtype=int),
                                         np.array(rel_list, dtype=int),
                                         np.array(tail_list, dtype=int))

            next_id += 1
        num_no_query_ent = 0
        num_one_query_ent = 0
        num_multiple_ent = 0
        for i in range(next_id):
            ct = num_query_entity[i]
            if ct == 1:
                num_one_query_ent += 1
            elif ct == 0:
                num_no_query_ent += 1
            else:
                num_multiple_ent += 1
        print("{} cases in total, {} cases without query entity, {} cases with single query entity,"
              " {} cases with multiple query entities".format(next_id, num_no_query_ent,
                                                              num_one_query_ent, num_multiple_ent))

    def _load_num(self):
        print('converting global to local entity index ...')
        self.local_entities_num = np.full((self.num_data, self.num_entity_threshold), len(self.entity2id), dtype=int)
        self.q2e_adj_mats_num = np.zeros((self.num_data, self.num_entity_threshold, 1), dtype=float)
        self.kb_fact_rels_num = np.full((self.num_data, self.num_fact_threshold), self.num_kb_relation, dtype=int)
        self.entity2fact_e_num = np.full((self.num_data, self.num_fact_threshold), -1, dtype=int)
        self.entity2fact_f_num = np.full((self.num_data, self.num_fact_threshold), -1, dtype=int)
        self.fact2entity_f_num = np.full((self.num_data, self.num_fact_threshold), -1, dtype=int)
        self.fact2entity_e_num = np.full((self.num_data, self.num_fact_threshold), -1, dtype=int)
        self.num_indices = np.full((self.num_data, 4, self.num_entity_threshold), -1, dtype=int)
        self.local_num_indices = np.full((self.num_data, 4, self.num_entity_threshold), -1, dtype=int)
        self.num_order = np.full((self.num_data, 4, self.num_entity_threshold), -1, dtype=int)
        self.numbers = np.full((self.num_data, 4, self.num_entity_threshold), -1, dtype=int)
        self.judge_question = np.zeros((self.num_data, 1), dtype=int)
        self.number_mask = np.zeros((self.num_data, self.num_entity_threshold), dtype=float)
        self.num_entity_mask = np.full((self.num_data, self.num_entity_threshold), len(self.entity2id), dtype=int)

        self.repeat_entity_map = np.full((self.num_data, 70, self.num_entity_threshold), -1, dtype=int)
        self.repeat_entity_index = np.full((self.num_data, 70, self.num_entity_threshold), -1, dtype=int)

        self.global2local_num_entity_maps = self._build_global2local_num_entity_maps()

        print('preparing num data ...')
        self._prepare_num()

    def judge_number_1(self, data):
        number = None

        if len(data.split('^^')) > 1:
            if len(data.split('^^')[0].split('"')) > 1:
                temp = data.split('^^')[0].split('"')[1]
            else:
                temp = data.split('^^')[0]

            tag = data.split('^^')[1].split('#')[1]
            if tag == 'date':
                if re.match('\d{4}[-/]\d{2}[-/]\d{2}', temp):
                    number = int(temp.replace('-', ''))


            elif tag == 'gYear':
                if temp[0] == '-':
                    if len(temp) > 8 and temp[5] == '-' and temp[8] == '-':
                        number = -1 * int(temp[1:].replace('-', ''))
                    elif len(temp) > 5 and temp[5] == '-':
                        number = -1 * int(temp[1:].replace('-', '') + '00')
                    else:
                        number = -1 * int(temp[1:] + '0000')
                elif re.match('\d{4}', temp):
                    number = int(temp + '0000')

            elif tag == 'gYearMonth':
                if re.match('\d{4}[-/]\d{2}', temp):
                    number = int(temp.replace('-', '') + '00')

            elif tag == 'dateTime':
                # print(data)
                temp = temp.replace('T', '.').replace('-', '').replace(':', '').replace('Z', '').replace('+', '')
                if temp[0] == '.':
                    number = float('0' + temp)
                else:
                    number = float(temp)

                # print(number)


        else:
            if len(data.split('"')) > 1:
                number = float(data.split('"')[1])
            else:
                number = float(data)

        # print(data,number)
        if number == None:
            print(data)
        return number

    def judge_number_2(self, data, rel):
        number = None
        if 'measurement_unit.dated_integer.number' in rel or  'measurement_unit.dated_percentage.number' in rel\
                or rel == 'location.statistical_region.part_time_employment_percent|measurement_unit.dated_percentage.rate':
            #print(rel)
            number = float(data)
        # elif rel == "soccer.football_player.statistics|soccer.football_player_stats.total_goals" or rel == "american_football.football_player.receiving|american_football.player_receiving_statistics.touchdowns":
        #     number = float(data)
        elif 'recurring_ceremony' in rel or 'opened' in rel or 'opened' in rel or 'date' in rel or rel.endswith(".to") \
                or 'established' in rel or 'founded' in rel or 'started' in rel or 'maiden_flight' in rel or '.from' in rel \
                or 'career_start' in rel or 'career_end' in rel or 'position_held' in rel or 'election_year' in rel:

            if data[0] == '-':
                if len(data) > 8 and data[5] == '-' and data[8] == '-':
                    number = -1 * int(data[1:].replace('-', ''))
                elif len(data) > 5 and data[5] == '-':
                    number = -1 * int(data[1:].replace('-', '') + '00')
                else:
                    number = -1 * int(data[1:] + '0000')

            elif 'T' in data or 'Z' in data:
                data = data.replace('T', '.').replace('-', '').replace(':', '').replace('Z', '').replace('+', '')
                if data[0] == '.':
                    number = float('0' + data)
                else:
                    number = float(data)

            elif re.match('\d{4}[-/]\d{2}[-/]\d{2}', data):
                number = int(data.replace('-', ''))

            elif re.match('\d{4}[-/]\d{2}', data):
                number = int(data.replace('-', '') + '00')

            elif re.match('\d{4}', data):
                number = int(data.replace('-', '') + '0000')

        else:
            number = float(data)
        if number == None:
            print(data)
        return number

    def judge_number(self, data, rel):
        if 'EMNLP_data' in self.data_folder:
            return self.judge_number_1(data)
        else:
            return self.judge_number_2(data, rel)

    def _prepare_num(self):
        next_id = 0
        for sample in tqdm(self.number_data):

            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            g2l_num = self.global2local_num_entity_maps[next_id]

            if len(g2l) == 0:
                print(next_id)
                continue

            if sample['id'] in self.cons_question:
                self.judge_question[next_id, 0] = 1

            for global_entity, local_entity in g2l_num.items():
                if local_entity != 0:
                    self.local_entities_num[next_id, local_entity] = global_entity

            i = len(self.data[next_id]['subgraph']['tuples'])

            for j, tpl in enumerate(sample['num_subgraphs']['tuples']):
                sbj, rel, obj = tpl
                if obj in g2l_num and sbj in g2l_num:
                    self.entity2fact_e_num[next_id, j] = g2l_num[obj]
                    self.entity2fact_f_num[next_id, j] = j
                    self.fact2entity_f_num[next_id, j] = j
                    self.fact2entity_e_num[next_id, j] = g2l_num[sbj]
                    self.kb_fact_rels_num[next_id, j] = rel

                if j >= self.num_fact_threshold - 1:
                    break

            for h, entity in enumerate(sample['num_subgraphs']['entities']):
                if entity in g2l_num:
                    if self.number_mask[next_id, g2l_num[entity]] == 1:
                        # self.q2e_adj_mats[next_id, g2l[self.entity2id[unicode(entity['text'])]], 0] = 1.0
                        self.q2e_adj_mats_num[next_id, g2l_num[entity], 0] = 1.0

            relation_map = defaultdict(set)
            repeat_entity = defaultdict(list)
            repeat_index = defaultdict(list)

            for a, tpl in enumerate(sample['num_subgraphs']['tuples']):
                sbj, rel, obj = tpl
                if obj in g2l_num and sbj in g2l_num:
                    e1 = self.id2entity[sbj]
                    rel_num = rel
                    num = self.id2entity[obj]
                    relation_map[rel_num].add(obj)
                    repeat_entity[(sbj, rel)].append(obj)
                    repeat_index[(sbj, rel)].append(i + a)

                if a >= self.num_fact_threshold - 1: break

            count = 0
            filter_rel_list = ['award.award_category.category_of', "film.performance.film",
                               "protected_sites.protected_site.annual_visitors"]
            for rel_id, num_indice in relation_map.items():
                if rel_id in self.expand_id2relation and self.expand_id2relation[rel_id] in filter_rel_list:
                    continue
                try:
                    number = []
                    for indice in num_indice:
                        try:
                            tp = self.judge_number(self.id2entity[indice], self.expand_id2relation[rel_id])
                            number.append(tp)
                        except:
                            print(self.id2entity[indice], self.expand_id2relation[rel_id])
                except:
                    print("Fail")
                    print(rel_id, num_indice)
                    print(self.expand_id2relation[rel_id])
                    for indice in num_indice:
                        print(self.id2entity[indice])
                    continue
                number_indices = [indice for indice in num_indice]
                local_num_indices = [g2l_num[indice] for indice in num_indice]
                num_len = len(number)
                if num_len > 0 and num_len <= self.num_entity_threshold:
                    if None in number:
                        print(self.id2rel[rel_id], number)
                    number_order = self.get_number_order(number)
                    self.num_indices[next_id, count, :num_len] = np.array(number_indices, dtype=int)
                    self.local_num_indices[next_id, count, :num_len] = np.array(local_num_indices, dtype=int)
                    self.num_order[next_id, count, :num_len] = np.array(number_order, dtype=int)
                    self.numbers[next_id, count, :num_len] = np.array(number, dtype=float)
                    count += 1

                if count >= 4:
                    break

            next_id += 1

    def get_number_order(self, numbers):
        if len(numbers) < 1:
            return None
        ordered_idx_list = np.argsort(np.array(numbers)).tolist()

        rank = 0
        number_rank = []
        for i, idx in enumerate(ordered_idx_list):
            if i == 0 or numbers[ordered_idx_list[i]] != numbers[ordered_idx_list[i - 1]]:
                rank += 1
            number_rank.append(rank)

        ordered_idx_rank = zip(ordered_idx_list, number_rank)

        final_rank = sorted(ordered_idx_rank, key=lambda x: x[0])
        final_rank = [item[1] for item in final_rank]

        return final_rank

    def _build_query_graph_new(self, sample_ids):
        word_ids = np.array([], dtype=int)
        layer_heads = {}
        layer_tails = {}
        layer_map = {}
        root_pos = []
        for i, sample_id in enumerate(sample_ids):
            word_ids = np.append(word_ids, self.query_texts[sample_id, :])
            index_bias = i * self.max_query_word
            node_layer = self.node2layer[sample_id]
            parents = self.dep_parents[sample_id]
            for j, par in enumerate(parents):
                if par == -1:   # root node, par = -1, layer = 1
                    root_pos.append(index_bias + j)
                    continue
                cur_layer = node_layer[j]
                node_now = j + index_bias
                parent_node = par + index_bias
                layer_heads.setdefault(cur_layer - 1, [])
                layer_tails.setdefault(cur_layer - 1, [])
                layer_map.setdefault(cur_layer, {})
                layer_map.setdefault(cur_layer - 1, {})
                if node_now not in layer_map[cur_layer]:
                    layer_map[cur_layer][node_now] = len(layer_map[cur_layer])
                if parent_node not in layer_map[cur_layer - 1]:
                    layer_map[cur_layer - 1][parent_node] = len(layer_map[cur_layer - 1])
                layer_heads[cur_layer - 1].append(layer_map[cur_layer][node_now])
                layer_tails[cur_layer - 1].append(layer_map[cur_layer - 1][parent_node])
                if j not in parents:
                    # if node is leave node, add zero node from previous layer
                    layer_heads.setdefault(cur_layer, [])
                    layer_tails.setdefault(cur_layer, [])
                    layer_heads[cur_layer].append(0)
                    layer_tails[cur_layer].append(layer_map[cur_layer][node_now])
        max_layer = max(list(layer_heads.keys()))
        # organize data layer-wise
        edge_lists = []
        number_node_total = 1  # initial node zero vector
        word_order = [0] * (len(sample_ids) * self.max_query_word)
        for layer in range(max_layer, 0, -1):
            # 1 ~ max_layer
            num_node = len(layer_map[layer])
            id2node = {v: k for k, v in layer_map[layer].items()}
            layer_entities = []
            for id in range(num_node):
                batch_node_idx = id2node[id]
                layer_entities.append(word_ids[batch_node_idx])
                word_order[batch_node_idx] = id + number_node_total
            tp_heads = []
            for node in layer_heads[layer]:
                if node == 0:   # Further check, there may be bug
                    tp_heads.append(0)
                else:
                    # zero index for leaf node in degree
                    tp_heads.append(node + 1)
            tp_heads = np.array(tp_heads)
            number_node_total += num_node
            fact_ids = np.array(range(len(layer_heads[layer])), dtype=int)
            tp_tails = np.array(layer_tails[layer])  # + number_node_total
            edge_list = (tp_heads, None, tp_tails, fact_ids)
            edge_lists.append((edge_list, layer_entities))
        root_order = [word_order[item] for item in root_pos]
        return edge_lists, word_order, root_order

    def _build_number_mat(self, sample_ids):
        entity2fact_e_num = self.entity2fact_e_num[sample_ids]
        entity2fact_f_num = self.entity2fact_f_num[sample_ids]
        fact2entity_f_num = self.fact2entity_f_num[sample_ids]
        fact2entity_e_num = self.fact2entity_e_num[sample_ids]
        local_entities_num = self.local_entities_num[sample_ids]
        q2e_adj_mats_num = self.q2e_adj_mats_num[sample_ids]
        kb_fact_rels_num = self.kb_fact_rels_num[sample_ids]
        num_indices = self.num_indices[sample_ids]
        local_num_indices = self.local_num_indices[sample_ids]
        num_order = self.num_order[sample_ids]
        number_mask = self.number_mask[sample_ids]
        repeat_entity_map = self.repeat_entity_map[sample_ids]
        repeat_entity_index = self.repeat_entity_index[sample_ids]
        num_entity_mask = self.num_entity_mask[sample_ids]
        judge_question = self.judge_question[sample_ids]

        def build_kb_adj_mat(entity2fact_e, entity2fact_f, fact2entity_f, fact2entity_e):
            """Create sparse matrix representation for batched data"""
            mats0_batch = []
            mats0_0 = []
            mats0_1 = []
            vals0 = []

            mats1_batch = []
            mats1_0 = []
            mats1_1 = []
            vals1 = []

            for i in range(len(entity2fact_e)):
                mat0_0 = entity2fact_f[i][entity2fact_f[i] != -1]
                mat0_1 = entity2fact_e[i][entity2fact_e[i] != -1]
                mat1_0 = fact2entity_e[i][fact2entity_e[i] != -1]
                mat1_1 = fact2entity_f[i][fact2entity_f[i] != -1]
                val0 = np.array([1.0] * len(entity2fact_f[i][entity2fact_f[i] != -1]))
                val1 = np.array([1.0] * len(fact2entity_e[i][fact2entity_e[i] != -1]))

                assert len(val0) == len(val1)
                num_fact = len(val0)
                num_keep_fact = int(np.floor(num_fact))
                mask_index = np.random.permutation(num_fact)[: num_keep_fact]
                # mat0
                mats0_batch.append(np.full(len(mask_index), i, dtype=int).tolist())
                mats0_0.append(mat0_0[mask_index].tolist())
                mats0_1.append(mat0_1[mask_index].tolist())
                vals0.append(val0[mask_index].tolist())
                # mat1
                mats1_batch.append(np.full(len(mask_index), i, dtype=int).tolist())
                mats1_0.append(mat1_0[mask_index].tolist())
                mats1_1.append(mat1_1[mask_index].tolist())
                vals1.append(val1[mask_index].tolist())

            return mats0_batch, mats0_0, mats0_1, vals0, mats1_batch, mats1_0, mats1_1, vals1

        mats0_batch_num, mats0_0_num, mats0_1_num, vals0_num, mats1_batch_num, mats1_0_num, mats1_1_num, vals1_num = \
            build_kb_adj_mat(entity2fact_e_num, entity2fact_f_num, fact2entity_f_num, fact2entity_e_num)
        question_tokenized = self.question_token[sample_ids]
        attention_mask = self.question_attention_mask[sample_ids]
        return question_tokenized, attention_mask, local_entities_num, kb_fact_rels_num, q2e_adj_mats_num, num_indices,\
               num_order, repeat_entity_map, repeat_entity_index, number_mask, num_entity_mask, mats0_batch_num,\
               mats0_0_num, mats0_1_num, vals0_num, mats1_batch_num, mats1_0_num, mats1_1_num, vals1_num,\
               judge_question, local_num_indices

    def get_batch_question_types(self):
        sample_ids = self.sample_ids
        type_list = []
        for index in sample_ids:
            type_list.append(self.question_type_list[index])
        return type_list

    def _build_fact_mat(self, sample_ids, fact_dropout):
        batch_heads = np.array([], dtype=int)
        batch_rels = np.array([], dtype=int)
        batch_tails = np.array([], dtype=int)
        batch_ids = np.array([], dtype=int)
        for i, sample_id in enumerate(sample_ids):
            index_bias = i * self.max_local_entity
            head_list, rel_list, tail_list = self.kb_adj_mats[sample_id]
            num_fact = len(head_list)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]

            real_head_list = head_list[mask_index] + index_bias
            real_tail_list = tail_list[mask_index] + index_bias
            real_rel_list = rel_list[mask_index]
            batch_heads = np.append(batch_heads, real_head_list)
            batch_rels = np.append(batch_rels, real_rel_list)
            batch_tails = np.append(batch_tails, real_tail_list)
            batch_ids = np.append(batch_ids, np.full(len(mask_index), i, dtype=int))
            if self.use_self_loop:
                num_ent_now = len(self.global2local_entity_maps[sample_id])
                ent_array = np.array(range(num_ent_now), dtype=int) + index_bias
                rel_array = np.array([self.num_kb_relation - 1] * num_ent_now, dtype=int)
                batch_heads = np.append(batch_heads, ent_array)
                batch_tails = np.append(batch_tails, ent_array)
                batch_rels = np.append(batch_rels, rel_array)
                batch_ids = np.append(batch_ids, np.full(num_ent_now, i, dtype=int))
        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        head_count = Counter(batch_heads)
        weight_list = [1.0 / head_count[head] for head in batch_heads]
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        print('max local entity: ', self.max_local_entity)
        return global2local_entity_maps

    def _build_global2local_num_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps_num = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.number_data):
            g2l_num = dict()
            # construct a map from global entity id to local entity id
            g2l = self.global2local_entity_maps[next_id]
            self._add_num_entity_to_map(next_id, sample['num_subgraphs']['entities'], g2l, g2l_num)

            global2local_entity_maps_num[next_id] = g2l_num
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        print('max local entity: ', self.max_local_entity)
        return global2local_entity_maps_num

    def _add_num_entity_to_map(self, next_id, entities, g2l, g2l_num ):
        for ii, entity_global_id in enumerate(entities):
            if entity_global_id not in g2l_num:
                if entity_global_id in g2l:
                    self.num_entity_mask[next_id, len(g2l_num)] = g2l[entity_global_id]
                else:
                    self.number_mask[next_id, len(g2l_num)] = 1
                g2l_num[entity_global_id] = len(g2l_num)

            if ii >= self.num_entity_threshold_all-2:
                break

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity_global_id in entities:
            if entity_global_id not in g2l:
                g2l[entity_global_id] = len(g2l)

    def deal_q_type(self, q_type=None):
        sample_ids = self.sample_ids
        if q_type is None:
            q_type = self.q_type
        if q_type == "seq":
            q_input = self.query_texts[sample_ids]
        else:
            raise NotImplementedError
        return q_input
