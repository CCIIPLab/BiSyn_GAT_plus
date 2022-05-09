import argparse
import six
from vocab import Vocab
from dataloader import ABSA_Dataset, ABSA_DataLoader, ABSA_collate_fn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='../data/V2/MAMS')
    parser.add_argument('--vocab_dir', type=str, default='../data/V2/MAMS')

    parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')
    
    parser.add_argument('--rnn_hidden_dim', type=int, default=200, help='rnn hidden dim.')
    parser.add_argument("--rnn_layers", type=int, default=1, help="Number of rnn layers in aa module.")
    parser.add_argument('--bidirect', default=True, help='bidirectional rnn')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='bert dim.')
    parser.add_argument('--hidden_dim', type=int, default=200)

    parser.add_argument('--input_dropout', type=float, default=0.1, help='input dropout rate.')
    parser.add_argument('--layer_dropout', type=float, default=0.0, help='layer dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='self-attention layer dropout rate.')

    parser.add_argument('--lower', default=True, help = 'lowercase all words.')
    parser.add_argument('--need_preprocess', default=False, help = 'need parse data.')

    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate.')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='learning rate for bert.')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight decay rate.')

    parser.add_argument('--num_encoder_layer', type=int, default=3, help='Number of graph layers.')
    parser.add_argument('--num_epoch', type=int, default=20, help='Number of total training epochs.')
    parser.add_argument('--max_patience', type=int, default=20, help='max patience in training')
    parser.add_argument('--batch_size', type=int, default=12, help='Training batch size.')
    parser.add_argument('--log_step', type=int, default=16, help='Print log every k steps.')

    parser.add_argument('--seed', type=int, default= 1)
    parser.add_argument('--max_len', type=int, default=100)

    parser.add_argument('--average_mapback', default=False, action='store_true')

    parser.add_argument('--leaf2root', default=False, action='store_true')
    parser.add_argument('--root2leaf', default=False, action='store_true')

    parser.add_argument('--con_dep_version', default='con_dot_dep', type=str)
    parser.add_argument('--con_dep_conditional', default=False, action = 'store_true')

    parser.add_argument('--attn_head', type=int, default=2)
    parser.add_argument('--max_num_spans', type=int, default=3)
    parser.add_argument('--special_token', default='[N]')
    parser.add_argument('--adj_span_version', type=int, default=0)

    parser.add_argument('--aa_num_layer', type=int, default=6, help='control aa module')
    # aa_graph_version
    parser.add_argument('--aa_graph_version', type=int, default=1, help = 'for aspect graph')
    parser.add_argument('--aspect_graph_num_layer', default=2,  type=int,help='for aspect_graph')
    parser.add_argument('--aspect_graph_encoder_version', default=1, type=int, help='for aspect_graph')
    parser.add_argument('--aa_graph_self', default=True, action='store_true', help='for aspect graph')

    parser.add_argument('--split_aa_graph', default=False, action='store_true', help='split forward and backward')
    
    parser.add_argument('--plus_AA', default=False, action='store_true', help='if add AA module')  # * add
    parser.add_argument('--is_filtered', default=True, help='neighboring aspects pairs')  # * add
    parser.add_argument('--is_average', default=True, help='average word vector')
    parser.add_argument('--sort_key_idx',default=0, help='sort idx')
    parser.add_argument('--borrow_encoder', default=False, action='store_true', 
                        help='if inter-context module share the same context encoder with intra-context module')

    args = parser.parse_args()


    # args = custom_args(args)

    return args


def custom_args(args):
    args.data_dir = 'data/V2/MAMS'
    args.vocab_dir = 'data/V2/MAMS'
    args.batch_size = 32
    args.input_dropout = 0.2
    args.layer_dropout = 0.2
    args.attn_head = 2
    args.average_mapback = True 
    args.plus_AA = True 
    args.lr = 1e-5
    args.split_aa_graph = True
    args.aspect_graph_num_layer = 1
    args.con_dep_version = 'con_dot_dep'
    args.con_dep_conditional = True 
    args.seed = 12
    args.num_epoch = 20
    args.borrow_encoder = True # TEST

    return args



def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

def totally_parameters(model):  #
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def load_vocab(args):
    print('Loading vocab...')

    vocab = {
        'token': Vocab.load_vocab(args.vocab_dir + '/vocab_tok.vocab'),
        'polarity': Vocab.load_vocab(args.vocab_dir + '/vocab_pol.vocab')
    }

    print(
        'token_vocab: {}, polarity_vocab:{}'.format(len(vocab['token']), len(vocab['polarity']))
    )

    args.tok_size = len(vocab['token'])
    return vocab


def load_one_data(args, file_name, vocab, tokenizer, block_shuffle = True, is_shuffle=True):
    print('Loading data from {} with batch size {}...'.format(file_name, args.batch_size))
    one_dataset = ABSA_Dataset(args, file_name, vocab, tokenizer)

    if block_shuffle and is_shuffle:
        one_dataloader = ABSA_DataLoader(one_dataset, 
                                        sort_key = lambda x: x[args.sort_key_idx],
                                        is_shuffle = is_shuffle,
                                        batch_size = args.batch_size,
                                        collate_fn = ABSA_collate_fn
                                        )
    else:
        one_sampler = RandomSampler(one_dataset) if is_shuffle else SequentialSampler(one_dataset)

        one_dataloader = DataLoader(one_dataset, 
                                    sampler=one_sampler,
                                    batch_size=args.batch_size,
                                    collate_fn = ABSA_collate_fn)
    return one_dataloader

def load_data(args, vocab, tokenizer=None):
    train_dataloader = load_one_data(args, file_name = args.data_dir + '/train_con_new.json',
                                     vocab = vocab, tokenizer = tokenizer, is_shuffle = True)

    valid_dataloader = load_one_data(args, file_name = args.data_dir + '/valid_con_new.json',
                                     vocab = vocab, tokenizer = tokenizer, is_shuffle = False)

    test_dataloader = load_one_data(args, file_name = args.data_dir + '/test_con_new.json',
                                     vocab = vocab, tokenizer = tokenizer, is_shuffle = False)
    
    return train_dataloader, valid_dataloader, test_dataloader

