import torch
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import BertModel, BertConfig
from layer import H_TransformerEncoder


class BiSyn_GAT_plus(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_dim = args.bert_hidden_dim + args.hidden_dim 
        self.args = args 

        self.intra_context_module = Intra_context(args)

        # inter_context_module
        if args.plus_AA:
            self.inter_context_module = Inter_context(args)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_dim, args.num_class)
    
    def forward(self, inputs):
        length, bert_length, word_mapback, \
        adj, aa_graph, aa_graph_length, \
        map_AS, map_AS_idx, \
        bert_sequence, bert_segments_ids, \
        aspect_indi, \
        con_spans, \
        map_AA, map_AA_idx,\
        aa_choice_inner_bert, aa_choice_inner_bert_length = inputs

        # aspect-level
        Intra_context_input = (length[map_AS], bert_length[map_AS], word_mapback[map_AS], aspect_indi, bert_sequence, bert_segments_ids, adj[map_AS], con_spans)

        Intra_context_output = self.intra_context_module(Intra_context_input)

        # sentence-level
        if self.args.plus_AA and map_AA.numel(): # BiSyn-GAT+
            Inter_context_input = (aa_choice_inner_bert, aa_choice_inner_bert_length, 
                                    map_AA, map_AA_idx, map_AS, map_AS_idx, 
                                        aa_graph_length, aa_graph)
            # sentence-level to aspect-level
            hiddens = self.inter_context_module(as_features = Intra_context_output, 
                                                inputs = Inter_context_input, \
                                                context_encoder = self.intra_context_module.context_encoder if self.args.borrow_encoder else None)
            
        else: # BiSyn-GAT
            hiddens = Intra_context_output 

        # aspect-level
        logits = self.classifier(self.dropout(hiddens))
        return logits

# Intra-context module
class Intra_context(nn.Module):
    def __init__(self, args):
        super().__init__()
         
        self.args = args 

        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.output_hidden_states = True
        bert_config.num_labels = 3

        self.layer_drop = nn.Dropout(args.layer_dropout)
        self.context_encoder = BertModel.from_pretrained('bert-base-uncased',config=bert_config)
        self.dense = nn.Linear(args.bert_hidden_dim, args.hidden_dim)
        self.graph_encoder = H_TransformerEncoder(
            d_model = args.hidden_dim,
            nhead = args.attn_head,
            num_encoder_layers = args.num_encoder_layer,
            inner_encoder_layers = args.max_num_spans,
            dropout = args.layer_dropout,
            dim_feedforward = args.bert_hidden_dim,
            activation = 'relu',
            layer_norm_eps = 1e-5
        )

    def forward(self, inputs):

        length, bert_lengths,  word_mapback, mask, bert_sequence, bert_segments_ids, adj, con_spans = inputs

        ###############################################################
        # 1. contextual encoder
        bert_outputs = self.context_encoder(bert_sequence, token_type_ids=bert_segments_ids)

        bert_out, bert_pooler_out = bert_outputs.last_hidden_state, bert_outputs.pooler_output

        bert_out = self.layer_drop(bert_out)

        # rm [CLS] 
        bert_seq_indi = ~sequence_mask(bert_lengths).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_lengths) + 1, :] * bert_seq_indi.float()
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))

        # average
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)  

        ###############################################################
        # 2. graph encoder
        key_padding_mask = sequence_mask(length)  # [B, seq_len]

        # from phrase(span) to form mask
        B, N, L = con_spans.shape
        span_matrix = get_span_matrix_4D(con_spans.transpose(0, 1))

        if self.args.con_dep_version == 'con_add_dep':
            # adj + span
            adj_matrix = adj.unsqueeze(dim=0).repeat(N, 1, 1, 1)
            assert ((adj_matrix[0] != adj_matrix[1]).sum() == 0)
            span_matrix = (span_matrix + adj_matrix).bool()

        elif self.args.con_dep_version == 'wo_dep':
            # only span
            pass 
    
        elif self.args.con_dep_version == 'wo_con':
            # only adj
            adj_matrix = adj.unsqueeze(dim=0)
            span_matrix = adj_matrix.repeat(N, 1, 1, 1)

        elif self.args.con_dep_version == 'con_dot_dep':
            # adj * span
            adj_matrix = adj.unsqueeze(dim=0).repeat(N, 1, 1, 1)
            assert ((adj_matrix[0] != adj_matrix[1]).sum() == 0)
            span_matrix = (span_matrix * adj_matrix).bool()
        
        graph_out = self.graph_encoder(bert_out,
                                       mask=span_matrix, src_key_padding_mask=key_padding_mask)
        ###############################################################
        # 3. fusion
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h
        # h_t
        bert_enc_outputs = (bert_out * mask).sum(dim=1) / asp_wn

        # g_t
        graph_enc_outputs = (graph_out * mask).sum(dim=1) / asp_wn  # mask h

        as_features = torch.cat([graph_enc_outputs +  bert_enc_outputs, bert_pooler_out],-1)
        return as_features
        

# Inter-context module
class Inter_context(nn.Module):
    def __init__(self, args, sent_encoder=None):
        super().__init__()
        self.args = args 
        in_dim = args.bert_hidden_dim + args.hidden_dim
        self.layer_drop = nn.Dropout(args.layer_dropout)
        if not args.borrow_encoder:
            self.sent_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.dense = nn.Linear(args.bert_hidden_dim, args.hidden_dim)

        self.con_aspect_graph_encoder = H_TransformerEncoder(
                        d_model = in_dim,
                        nhead = args.attn_head,
                        num_encoder_layers = args.aspect_graph_num_layer,
                        inner_encoder_layers = args.aspect_graph_encoder_version,
                        dropout = args.layer_dropout,
                        dim_feedforward=args.hidden_dim,
                        activation = 'relu',
                        layer_norm_eps=1e-5
                    )

    def forward(self, as_features, inputs, context_encoder=None):
        aa_choice_inner_bert, aa_choice_inner_bert_length, \
            map_AA, map_AA_idx, map_AS, map_AS_idx, \
                aa_graph_length, aa_graph = inputs

        need_change = (aa_graph_length[map_AS] > 1)

        inner_v_node, inner_v = self.forward_bert_inner( aa_choice_inner_bert,
                                                        aa_choice_inner_bert_length,
                                                        context_encoder) 

        rela_v_inner = torch.cat((inner_v_node.sum(dim=1), inner_v), dim=-1)
        AA_features = self.con_aspect_graph(rela_v_inner, 
                                            as_features, 
                                            map_AA, map_AA_idx, 
                                            map_AS, map_AS_idx,
                                            aa_graph_length, aa_graph)



        AA_features = AA_features * need_change.unsqueeze(dim=-1) + as_features * ~(need_change).unsqueeze(dim=-1)

        fusion_features = AA_features + as_features
        
        return fusion_features



    def forward_bert_inner(self, aa_choice_inner_bert, aa_choice_inner_bert_length, context_encoder = None):
                
        bert_outputs = self.sent_encoder(aa_choice_inner_bert) if context_encoder is None else context_encoder(aa_choice_inner_bert)
        bert_out, bert_pool_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        bert_out = self.layer_drop(bert_out)
        # rm [CLS] representation
        bert_seq_indi = ~sequence_mask(aa_choice_inner_bert_length).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(aa_choice_inner_bert_length) + 1, :] * bert_seq_indi.float()

        inner_v_node = self.dense(bert_out)
        inner_v = bert_pool_output
        return inner_v_node, inner_v


    def con_aspect_graph(self, 
                        rela_v, 
                        as_features, 
                        map_AA, map_AA_idx, map_AS, map_AS_idx, 
                        aa_graph_length, aa_graph):
        B = map_AS.max() + 1
        L = map_AA_idx.max() + 1
        graph_input_features = torch.zeros((B, L, as_features.shape[-1]), device=as_features.device)

        graph_input_features[map_AA, map_AA_idx] = rela_v
        graph_input_features[map_AS, map_AS_idx] = as_features


        aa_graph_key_padding_mask = sequence_mask(aa_graph_length)
        
        if self.args.aspect_graph_encoder_version == 1:
            # split and share parameters
            forward_ = self.con_aspect_graph_encoder(graph_input_features,
                                                    mask=aa_graph.unsqueeze(0),
                                                    src_key_padding_mask=aa_graph_key_padding_mask)
        
            backward_ = self.con_aspect_graph_encoder(graph_input_features,
                                                    mask=aa_graph.transpose(1, 2).unsqueeze(0),
                                                    src_key_padding_mask=aa_graph_key_padding_mask)
            mutual_influence = forward_ + backward_
        
        elif self.args.aspect_graph_encoder_version == 2:
            # not split    
            mutual_influence = self.con_aspect_graph_encoder(
                graph_input_features,
                mask = torch.cat((aa_graph.unsqueeze(dim=0), aa_graph.transpose(1,2).unsqueeze(dim=0)),dim=0),
                src_key_padding_mask = aa_graph_key_padding_mask
            )
        return mutual_influence[map_AS, map_AS_idx]


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))


def get_span_matrix_4D(span_list, rm_loop=False, max_len=None):
    '''
    span_list: [N,B,L]
    return span:[N,B,L,L]
    '''
    # [N,B,L]
    N, B, L = span_list.shape
    span = get_span_matrix_3D(span_list.contiguous().view(-1, L), rm_loop, max_len).contiguous().view(N, B, L, L)
    return span


def get_span_matrix_3D(span_list, rm_loop=False, max_len=None):
    # [N,L]
    origin_dim = len(span_list.shape)
    if origin_dim == 1:  # [L]
        span_list = span_list.unsqueeze(dim=0)
    N, L = span_list.shape
    if max_len is not None:
        L = min(L, max_len)
        span_list = span_list[:, :L]
    span = span_list.unsqueeze(dim=-1).repeat(1, 1, L)
    span = span * (span.transpose(-1, -2) == span)
    if rm_loop:
        span = span * (~torch.eye(L).bool()).unsqueeze(dim=0).repeat(N, 1, 1)
        span = span.squeeze(dim=0) if origin_dim == 1 else span  # [N,L,L]
    return span