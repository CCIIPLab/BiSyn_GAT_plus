import os 
import json
import torch 
import numpy as np 
from transformers import BertTokenizer

import copy 
import random 
import itertools 
from itertools import chain

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from spans import *

class ABSA_Dataset(Dataset):
    def __init__(self, args, file_name, vocab, tokenizer):
        super().__init__()

        # load raw data
        with open(file_name,'r',encoding='utf-8') as f:
            raw_data = json.load(f)

            if args.need_preprocess:
                raw_data = self.process_raw(raw_data)
                new_file_name = file_name.replace('.json','_con.json')
                with open(new_file_name, 'w', encoding='utf-8') as f:
                    json.dump(raw_data,f)
                print('Saving to:', new_file_name)

        self.data = self.process(raw_data, vocab, args, tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    
    def process_raw(self, data):
        # get parserd data
        # we already provide here
        pass
    

    def process(self, data, vocab, args, tokenizer):
        token_vocab = vocab['token']
        pol_vocab = vocab['polarity']

        processed = []
        max_len = args.max_len 
        CLS_id = tokenizer.convert_tokens_to_ids(["[CLS]"])
        SEP_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
        sub_len = len(args.special_token)


        for d in data:
            tok = list(d['token'])
            if args.lower:
                tok = [t.lower() for t in tok]
            
            text_raw_bert_indices, word_mapback, _ = text2bert_id(tok, tokenizer)

            text_raw_bert_indices = text_raw_bert_indices[:max_len]
            word_mapback = word_mapback[:max_len]

            length = word_mapback[-1] + 1

            # tok = tok[:length]
            bert_length = len(word_mapback)

            dep_head = list(d['dep_head'])[:length]

            # map2id 
            # tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
            
            # con
            con_head = d['con_head']
            con_mapnode = d['con_mapnode']
            con_path_dict, con_children = get_path_and_children_dict(con_head)
            mapback = [ idx for idx ,word in enumerate(con_mapnode) if word[-sub_len: ]!= args.special_token]

            layers, influence_range, node2layerid = form_layers_and_influence_range(con_path_dict, mapback)

            spans = form_spans(layers, influence_range, length, con_mapnode)

            adj_i_oneshot = head_to_adj_oneshot(dep_head, length, d['aspects'])

            cd_adj = np.ones((length,length))
            if args.con_dep_conditional:
                father = 1
                if father in con_children and [con_mapnode[node] for node in con_children[father]].count('S[N]') > 1 and con_mapnode[father] == 'S[N]':
                    cd_span = spans[node2layerid[father]+1]
                    cd_adj = get_conditional_adj(father, length, cd_span, con_children, con_mapnode)

            adj_i_oneshot = adj_i_oneshot * cd_adj 
            
            # aspect-specific
            bert_sequence_list = []
            bert_segments_ids_list = []
            label_list = []
            aspect_indi_list = []

            select_spans_list = []

            for aspect in d['aspects']:
                asp = list(aspect['term'])
                asp_bert_ids, _, _ = text2bert_id(asp, tokenizer)
                bert_sequence = CLS_id  + text_raw_bert_indices +  SEP_id + asp_bert_ids + SEP_id
                bert_segments_ids = [0] * (bert_length + 2) + [1] * (len(asp_bert_ids ) +1)

                bert_sequence = bert_sequence[:max_len+3]
                bert_segments_ids = bert_segments_ids[:max_len+3]

                label = aspect['polarity']

                aspect_indi = [0] * length 

                for pidx in range(aspect['from'], aspect['to']):
                    aspect_indi[pidx] = 1
                
                label = pol_vocab.stoi.get(label)

                aspect_range = list(range(mapback[aspect['from']], mapback[aspect['to']-1] + 1))

                con_lca = find_inner_LCA(con_path_dict, aspect_range)

                select_spans, span_indications = form_aspect_related_spans(con_lca, spans, con_mapnode, node2layerid, con_path_dict)

                select_spans = select_func(select_spans, args.max_num_spans, length)

                select_spans = [[ x+ 1 for x in span] for span in select_spans] 


                label_list.append(label)
                aspect_indi_list.append(aspect_indi)
                bert_sequence_list.append(bert_sequence)
                bert_segments_ids_list.append(bert_segments_ids)

                select_spans_list.append(select_spans)
            
            # aspect-aspect
            choice_list = [(idx, idx + 1) for idx in range(len(d['aspects']) - 1)] if args.is_filtered else list(
                itertools.combinations(list(range(len(d['aspects']))), 2))


            aa_choice_inner_bert_id_list = []
            
            
            num_aspects = len(d['aspects'])

            cnum = num_aspects  + len(choice_list)
            aa_graph = np.zeros((cnum, cnum))
            cnt = 0

            for aa_info in d['aa_choice']:
                select_ = (aa_info['select_idx'][0], aa_info['select_idx'][1])

                if select_ in choice_list: # choicen
                    first = aa_info['select_idx'][0]
                    second = aa_info['select_idx'][1]

                    word_range = aa_info['word_range']
                    select_words = d['token'][word_range[0]:word_range[-1] + 1] if (word_range[0] <= word_range[-1]) else ['and']

                    aa_raw_bert_ids, _, _ = text2bert_id(select_words, tokenizer)
                    aa_raw_bert_ids = aa_raw_bert_ids[:max_len]

                    if args.aa_graph_version == 1: #directional
                        if first % 2 == 0:
                            aa_graph[cnt + num_aspects][first] = 1 
                            aa_graph[second][cnt + num_aspects] = 1
                        else:
                            aa_graph[cnt + num_aspects][second] = 1
                            aa_graph[first][cnt + num_aspects] = 1
                    else: # undirectional
                        aa_graph[cnt + num_aspects][first] = 1
                        aa_graph[second][cnt + num_aspects] = 1
                    

                    if args.aa_graph_self:
                        aa_graph[first][first] = 1 
                        aa_graph[second][second] = 1
                        aa_graph[cnt + num_aspects][cnt + num_aspects] = 1 
                    

                    aa_choice_inner_bert_id_list.append(CLS_id + aa_raw_bert_ids + SEP_id)
                    
                    cnt += 1

            processed += [
                (
                    length, bert_length, word_mapback,
                    adj_i_oneshot, aa_graph,
                    # aspect-specific
                    bert_sequence_list, bert_segments_ids_list, aspect_indi_list, select_spans_list,
                    # aspect-aspect
                    aa_choice_inner_bert_id_list, 
                    # label
                    label_list
                )
            ]
        
        return processed 
                    

def ABSA_collate_fn(batch):
    batch_size = len(batch)
    batch = list(zip(*batch))

    lens = batch[0]

    (length_, bert_length_, word_mapback_,
    adj_i_oneshot_, aa_graph_,
    bert_sequence_list_, bert_segments_ids_list_, 
    aspect_indi_list_, select_spans_list_,
    aa_choice_inner_bert_id_list_, 
    label_list_) = batch 

    max_lens = max(lens)

    
    length = torch.LongTensor(length_)
    bert_length = torch.LongTensor(bert_length_)
    word_mapback = get_long_tensor(word_mapback_, batch_size)

    adj_oneshot = np.zeros((batch_size, max_lens, max_lens), dtype=np.float32)

    for idx in range(batch_size):
        mlen = adj_i_oneshot_[idx].shape[0]
        adj_oneshot[idx,:mlen,:mlen] = adj_i_oneshot_[idx]
    
    adj_oneshot = torch.FloatTensor(adj_oneshot)


    # Intra-context 
    map_AS = [[idx] * len(a_i) for idx, a_i in enumerate(bert_sequence_list_)]
    map_AS_idx = [range(len(a_i)) for a_i in bert_sequence_list_]

    # add_pre = np.array([0] + [len(m) for m in map_AS[:-1]]).cumsum()
    
    map_AS = torch.LongTensor([m for m_list in map_AS for m in m_list])
    map_AS_idx = torch.LongTensor([m for m_list in map_AS_idx for m in m_list])

    as_batch_size = len(map_AS)

    bert_sequence = [p for p_list in bert_sequence_list_ for p in p_list]
    bert_sequence = get_long_tensor(bert_sequence, as_batch_size)

    bert_segments_ids = [p for p_list in bert_segments_ids_list_ for p in p_list]
    bert_segments_ids = get_long_tensor(bert_segments_ids, as_batch_size)

    aspect_indi = [p for p_list in aspect_indi_list_ for p in p_list]
    aspect_indi = get_long_tensor(aspect_indi, as_batch_size)

   
    con_spans_list = [p for p_list in select_spans_list_ for p in p_list]
    max_num_spans = max([len(p) for p in con_spans_list])
    con_spans = np.zeros((as_batch_size, max_num_spans, max_lens), dtype=np.int64)
    for idx in range(as_batch_size):
        mlen = len(con_spans_list[idx][0])
        con_spans[idx,:,:mlen] = con_spans_list[idx]
    
    con_spans = torch.LongTensor(con_spans)

    # label
    label = torch.LongTensor([sl for sl_list in label_list_ for sl in sl_list if isinstance(sl, int)])

    # aa_graph
    aspect_num = [len(a_i) for a_i in bert_sequence_list_]
    max_aspect_num = max(aspect_num)

    if (max_aspect_num > 1):
        aa_graph_length = torch.LongTensor([2 * num - 1 for num in aspect_num])  # 相当于length
        aa_graph = np.zeros((batch_size, 2 * max_aspect_num - 1, 2 * max_aspect_num - 1))
        
        for idx in range(batch_size):
            cnum = aa_graph_length[idx]
            aa_graph[idx, :cnum, :cnum] = aa_graph_[idx]
        aa_graph = torch.LongTensor(aa_graph)
    else:
        aa_graph_length = torch.LongTensor([])
        aa_graph = torch.LongTensor([])

    aa_choice = [m for m_list in aa_choice_inner_bert_id_list_ for m in m_list]
    aa_batch_size = len(aa_choice)

    if aa_batch_size > 0:
        map_AA = [[idx] * len(a_i) for idx, a_i in enumerate(aa_choice_inner_bert_id_list_)]
        map_AA = torch.LongTensor([m for m_list in map_AA for m in m_list])

        map_AA_idx = torch.LongTensor([m + len(a_i) + 1 for a_i in aa_choice_inner_bert_id_list_ for m in range(len(a_i))])


        aa_choice_inner_bert_id = [m for m_list in aa_choice_inner_bert_id_list_ for m in m_list if len(m) > 0]
        aa_choice_inner_bert_length = torch.LongTensor([len(m) - 2 for m in aa_choice_inner_bert_id])
        aa_choice_inner_bert_id = get_long_tensor(aa_choice_inner_bert_id, aa_batch_size)


    else:
        map_AA = torch.LongTensor([])
        map_AA_idx = torch.LongTensor([])
       
        aa_choice_inner_bert_id = torch.LongTensor([])
        aa_choice_inner_bert_length = torch.LongTensor([])

    
    return (
        length, bert_length, word_mapback, adj_oneshot,
        aa_graph, aa_graph_length,
        map_AS, map_AS_idx,
        bert_sequence, bert_segments_ids,
        aspect_indi, con_spans,
        map_AA, map_AA_idx,
        aa_choice_inner_bert_id, aa_choice_inner_bert_length,
        label
    )


def text2bert_id(token, tokenizer):
    re_token = []
    word_mapback = []
    word_split_len = []
    for idx, word in enumerate(token):
        temp = tokenizer.tokenize(word)
        re_token.extend(temp)
        word_mapback.extend([idx] * len(temp))
        word_split_len.append(len(temp))
    re_id = tokenizer.convert_tokens_to_ids(re_token)
    return re_id ,word_mapback, word_split_len

class ABSA_DataLoader(DataLoader):
    def __init__(self, dataset, sort_key, sort_bs_num=None, is_shuffle=True, **kwargs):
        '''
        :param dataset: Dataset object
        :param sort_idx: sort_function
        :param sort_bs_num: sort range; default is None(sort for all sequence)
        :param is_shuffle: shuffle chunk , default if True
        :return:
        '''
        assert isinstance(dataset.data, list)
        super().__init__(dataset,**kwargs)
        self.sort_key = sort_key
        self.sort_bs_num = sort_bs_num
        self.is_shuffle = is_shuffle

    def __iter__(self):
        if self.is_shuffle:
            self.dataset.data = self.block_shuffle(self.dataset.data, self.batch_size, self.sort_bs_num, self.sort_key, self.is_shuffle)

        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, is_shuffle):
        # sort
        random.shuffle(data)
        data = sorted(data, key = sort_key) # 先按照长度排序
        batch_data = [data[i : i + batch_size] for i in range(0,len(data),batch_size)]
        batch_data = [sorted(batch, key = sort_key) for batch in batch_data]
        if is_shuffle:
            random.shuffle(batch_data)
        batch_data = list(chain(*batch_data))
        return batch_data

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens


def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.FloatTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
