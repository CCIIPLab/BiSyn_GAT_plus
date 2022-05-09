import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
import copy


class H_TransformerEncoder(nn.Module):
    '''
    Transformer Encoder
    '''

    def __init__(self, d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 inner_encoder_layers = 3,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-5):

        super(H_TransformerEncoder, self).__init__()
        encoder_layer = H_TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                inner_layer=inner_encoder_layers,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                layer_norm_eps=layer_norm_eps)

        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.num_layers = num_encoder_layers
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        '''
        Initiate parameters in the transformer model
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, src_key_padding_mask): #[表示相同操作重复num_encoder_layer次]
        # src:[bs,S,E]
        B, L, _ = src.shape
        output = src.transpose(0, 1)  # src:[S,bs,E]

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.encoder_norm is not None:
            output = self.encoder_norm(output)

        return output.transpose(0, 1)  # [bs,S,E]

class H_TransformerEncoderLayer(nn.Module): 
    def __init__(self, d_model, nhead, inner_layer = 3, dim_feedforward=2048, dropout=0.1, activation='relu',layer_norm_eps=1e-5):
        super(H_TransformerEncoderLayer, self).__init__()
        self.nhead = nhead
        inner_encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                layer_norm_eps=layer_norm_eps)
        self.layers = _get_clones(inner_encoder_layer, inner_layer)
        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src, src_mask, src_key_padding_mask,  isindi=True):

        output = src
        L,B,D = src.shape

        for idx, layer in enumerate(self.layers):
            
            # mask需要改成多头的
            mask_indi = src_mask[idx].bool().int() if isindi else ~(src_mask[idx].bool()).int()
            rm_inf = (mask_indi.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, mask_indi.shape[-1])
            attn_mask = mask_indi.float().masked_fill(mask_indi == 0, float('-inf')).masked_fill(mask_indi > 0, float(0.0)).masked_fill(rm_inf, float(0.0))
            attn_mask = torch.stack([attn_mask for _ in range(self.nhead)], dim=1).contiguous().view(-1, L, L)

            output = layer(output, src_mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
            
            assert(torch.isnan(output).sum() == 0)
        
        if self.encoder_norm is not None:
            output = self.encoder_norm(output)
            return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, 
                 dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout) 

        self.dropout = nn.Dropout(dropout)
        self.feedforword = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation),
            self.dropout,
            nn.Linear(dim_feedforward, d_model),
            self.dropout
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src_add_norm = self.norm(src + self.dropout(src2))
        return self.norm(src + self.feedforword(src_add_norm))

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.kdim)
        self.v_proj = nn.Linear(embed_dim, self.vdim)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        
        L, B, D = query.size()
        single_attn_mask = attn_mask.contiguous().view(B, -1, L, L)[:, 0, :, :]
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q * scaling

        # check attn_mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [B * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(L, B * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, B * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, B * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == B
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q,
                                        k.transpose(1, 2))  # [B*num_heads,L,D] * [B*num_heads,D,L] -->[B*num_heads,L,L]
        assert list(attn_output_weights.size()) == [B * self.num_heads, L, src_len]

            

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask


        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(B, self.num_heads, L, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B,N,L,L]->[B,1,1,L]
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(B * self.num_heads, L, src_len)
            
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

       

        attn_output = torch.bmm(attn_output_weights, v)  # [B,N,L,L] [B,N,L,D]
        assert list(attn_output.size()) == [B * self.num_heads, L, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, B, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(B, self.num_heads, L, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None