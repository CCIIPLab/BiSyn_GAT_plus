import os
import json
from collections import Counter
import itertools




def GetTree_heads(t):
    heads = [0] * len(t)
    mapnode = [0] * len(t)

    def Findheads(cidx, t, headidx):
        if (cidx >= len(t)):
            return cidx
        mapnode[cidx] = t[cidx].lhs()
        heads[cidx] = headidx

        if t[cidx].lhs().__str__() == '_':
            mapnode[cidx] = t[cidx].rhs()[0]

            return cidx + 1

        nidx = cidx + 1
        for r in t[cidx].rhs():
            nidx = Findheads(nidx, t, cidx)

        return nidx

    Findheads(0, t, -1)
    return heads, mapnode




def get_path_and_children_dict(heads):
    path_dict = {}
    remain_nodes = list(range(len(heads)))
    delete_nodes = []

    while len(remain_nodes) > 0:
        for idx in remain_nodes:
            # 初始状态
            if idx not in path_dict:
                path_dict[idx] = [heads[idx]]  # no self
                if heads[idx] == -1:
                    delete_nodes.append(idx)  # need delete root
            else:
                last_node = path_dict[idx][-1]
                if last_node not in remain_nodes:
                    path_dict[idx].extend(path_dict[last_node])
                    delete_nodes.append(idx)
                else:
                    path_dict[idx].append(heads[last_node])
        # remove nodes
        for del_node in delete_nodes:
            remain_nodes.remove(del_node)
        delete_nodes = []

    # children_dict
    children_dict = {}
    for x, l in path_dict.items():
        if l[0] == -1:
            continue
        if l[0] not in children_dict:
            children_dict[l[0]] = [x]
        else:
            children_dict[l[0]].append(x)

    return path_dict, children_dict


def find_inner_LCA(path_dict, aspect_range):
    path_range = [[x] + path_dict[x] for x in aspect_range]
    path_range.sort(key=lambda l: len(l))

    for idx in range(len(path_range[0])):
        flag = True
        for pid in range(1, len(path_range)):
            if path_range[0][idx] not in path_range[pid]:
                flag = False 
                break

        if flag:  
            LCA_node = path_range[0][idx]
            break  # already find
    return LCA_node

# get_word_range
def find_LCA_and_PATH(A, B):
    for idx in range(min(len(A), len(B))):
        if A[idx] in B:
            return A[idx], A[:idx], B[:B.index(A[idx])]
        elif B[idx] in A:
            return B[idx], A[:A.index(B[idx])], B[:idx]
    return -1, A[:-1], B[:-1]

def FindS(l, children, mapback):
    def inner_Find(x, index):
        if x[index] not in children:
            return x[index]
        else:
            return inner_Find(children[x[index]], index)

    return mapback.index(inner_Find(l, 0)), mapback.index(inner_Find(l, -1))

def get_word_range(lca_A, lca_B, path_dict, children, mapback, default_range): 
    
    LCA, pathA, pathB = find_LCA_and_PATH([lca_A] + path_dict[lca_A], [lca_B] + path_dict[lca_B])
    inner_node_LCA = children[LCA][children[LCA].index(pathA[-1]) + 1:children[LCA].index(pathB[-1])] if (
                len(pathA) and len(pathB)) else []
    word_range = FindS(inner_node_LCA, children, mapback) if len(inner_node_LCA) > 0 else \
        default_range  
    return word_range





def preprocess_file(file_name, dep_parser=None, con_parser=None, special_token='[N]'):
    
    
    print('Processing:',file_name)
    from tqdm import tqdm
    from supar import Parser
    if dep_parser is None:
        dep_parser = Parser.load('biaffine-dep-en')
    if con_parser is None:
        con_parser = Parser.load('crf-con-en')
        
    
    sub_len = len(special_token)
    
    with open(file_name,'r',encoding='utf-8') as f:
        data = json.load(f)
        
    for d in tqdm(data):
        token = d['token']
        token = [tok.replace(u'\xa0', u'') for tok in token]
        d['token'] = token
        
        # dependency parsing
        dataset = dep_parser.predict(token, verbose=False)
        dep_head = dataset.arcs[0]
        d['dep_head'] = [x-1 for x in dep_head]


        # constituent parsing
        parser_inputs = ' '.join(token).replace('(', '<').replace(')', '>').split(' ')  # [ver1]
        # parser_inputs = ' '.join(token).replace('(','<').replace(')','>').replace(r"'s",'is').replace(r"n't",'not').split(' ') #[ver2]
        dataset = con_parser.predict(parser_inputs, verbose=False)
        t = dataset.trees[0]
        con_head, con_mapnode = GetTree_heads(t.productions())
        d['con_head'] = con_head

        
        con_mapnode = [x if isinstance(x, str) else x.__str__() + special_token for x in con_mapnode]
        d['con_mapnode'] = con_mapnode
        
        
        d['aspects'].sort(key=lambda x:(x['to'],x['from']))

        con_path_dict,con_children = get_path_and_children_dict(d['con_head'])
        
        mapS = [
            idx for idx, word in enumerate(con_mapnode) if word[-sub_len:] == special_token and word[:-sub_len] == 'S'
        ]
        
        mapback = [ idx for idx,word in enumerate(con_mapnode) if word[-sub_len:]!=special_token]
        
        for aspect_info in d['aspects']:
            aspect_range = list(range(mapback[aspect_info['from']],mapback[aspect_info['to']-1]+1))

            con_lca = find_inner_LCA(con_path_dict, aspect_range)
            aspect_info['con_lca']  = con_lca

            
        choice_list = itertools.combinations(list(range(len(d['aspects']))),2)
        aa_choice = []
        for first,second in choice_list:
            temp = {'select_idx':(first,second)}
            A_asp = d['aspects'][first]
            B_asp = d['aspects'][second]

            default_range = (A_asp['to'],B_asp['from']-1)
            
            word_range = get_word_range(A_asp['con_lca'],
                                        B_asp['con_lca'],
                                        con_path_dict,
                                        con_children,mapback,
                                        default_range)


            assert(word_range[0] < len(token) and word_range[1] < len(token))
            
            temp['word_range'] = word_range
            temp['polarity_pair'] = (A_asp['polarity'],B_asp['polarity'])
            
            aa_choice.append(temp)
            
        d['aa_choice'] = aa_choice
        
    with open(file_name.replace('.json','_new.json'), 'w', encoding='utf-8') as f:
        json.dump(data,f)

    print('Done!')

if __name__ == '__main__':
    data_dir = 'data/V2'
    for data_set in ['Laptops','MAMS','Restaurants','Tweets']:
        for file_type in ['train','valid','test']:
            file_name = data_dir + '/' + data_set + '/' + file_type + '_con_new.json'
            preprocess_file(file_name)