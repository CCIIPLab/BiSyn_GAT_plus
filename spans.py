import numpy as np

def find_inner_LCA(path_dict,aspect_range):
    path_range = [ [x] + path_dict[x] for x in aspect_range]
    path_range.sort(key=lambda l:len(l))
 
    for idx in range(len(path_range[0])):
        flag = True
        for pid in range(1,len(path_range)):
            if path_range[0][idx]  not in path_range[pid]:
                flag = False #其中一个不在
                break
            
        if flag: #都在
            LCA_node = path_range[0][idx]
            break #already find
    return LCA_node

def get_path_and_children_dict(heads):
    path_dict = {}
    remain_nodes = list(range(len(heads)))
    delete_nodes = []
    
    while len(remain_nodes) > 0:
        for idx in remain_nodes:
            #初始状态
            if idx not in path_dict:
                path_dict[idx] = [heads[idx]]  # no self
                if heads[idx] == -1:
                    delete_nodes.append(idx) #need delete root
            else:
                last_node = path_dict[idx][-1]
                if last_node not in remain_nodes:
                    path_dict[idx].extend(path_dict[last_node])
                    delete_nodes.append(idx)
                else:
                    path_dict[idx].append(heads[last_node])
        #remove nodes
        for del_node in delete_nodes:
            remain_nodes.remove(del_node)
        delete_nodes = []

    #children_dict
    children_dict = {}
    for x,l in path_dict.items():
        if l[0] == -1:
            continue
        if l[0] not in children_dict:
            children_dict[l[0]] = [x]
        else:
            children_dict[l[0]].append(x)

    return path_dict, children_dict

def form_layers_and_influence_range(path_dict,mapback):
    sorted_path_dict = sorted(path_dict.items(),key=lambda x: len(x[1]))
    influence_range = { cid:[idx,idx+1] for idx,cid in enumerate(mapback) }
    layers = {}
    node2layerid = {}
    for cid,path_dict in sorted_path_dict[::-1]:
    
        length = len(path_dict)-1
        if length not in layers:
            layers[length] = [cid]
            node2layerid[cid] = length
        else:
            layers[length].append(cid)
            node2layerid[cid] = length
        father_idx = path_dict[0]
        
        
        assert(father_idx not in mapback)
        if father_idx not in influence_range:
            influence_range[father_idx] = influence_range[cid][:] #deep copy
        else:
            influence_range[father_idx][0] = min(influence_range[father_idx][0], influence_range[cid][0])
            influence_range[father_idx][1] = max(influence_range[father_idx][1], influence_range[cid][1])  
    
    layers = sorted(layers.items(),key=lambda x:x[0])
    layers = [(cid,sorted(l)) for cid,l in layers]  # or [(cid,l.sort()) for cid,l in layers]

    return layers, influence_range,node2layerid

def form_spans(layers, influence_range, token_len, con_mapnode, special_token = '[N]'):
    spans = []
    sub_len = len(special_token)
    
    for _, nodes in layers:

        pointer = 0
        add_pre = 0
        temp = [0] * token_len
        temp_indi = ['-'] * token_len
        
        for node_idx in nodes:
            begin,end = influence_range[node_idx] 
            
            if con_mapnode[node_idx][-sub_len:] == special_token:
                temp_indi[begin:end] = [con_mapnode[node_idx][:-sub_len]] * (end-begin)
            
            if(begin != pointer): 
                sub_pre = spans[-1][pointer] 
                temp[pointer:begin] = [x + add_pre-sub_pre for x in spans[-1][pointer:begin]] #
                add_pre = temp[begin-1] + 1
            temp[begin:end] = [add_pre] * (end-begin)  

            add_pre += 1
            pointer = end
        if pointer != token_len: 
            sub_pre = spans[-1][pointer]
            temp[pointer:token_len] = [x + add_pre-sub_pre for x in spans[-1][pointer:token_len]]
            add_pre = temp[begin-1] + 1
        spans.append(temp)

    return spans

def head_to_adj_oneshot(heads, sent_len, aspect_dict, 
                        leaf2root=True, root2leaf=True, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)

    heads = heads[:sent_len]

    # aspect <self-loop>
    for asp in aspect_dict:
        from_ = asp['from']
        to_ = asp['to']
        for i_idx in range(from_, to_):
            for j_idx in range(from_, to_):
                adj_matrix[i_idx][j_idx] = 1



    for idx, head in enumerate(heads):
        if head != -1:
            if leaf2root:
                adj_matrix[head, idx] = 1
            if root2leaf:
                adj_matrix[idx, head] = 1

        if self_loop:
            adj_matrix[idx, idx] = 1

    return adj_matrix

def get_conditional_adj(father, length, cd_span, 
                        con_children, con_mapnode):
    s_slist = [idx for idx, node in enumerate(con_children[father]) if con_mapnode[node] == 'S[N]' ]
    st_adj = np.ones((length,length))
    for i in range(len(s_slist)-1):
        idx = s_slist[i]
        begin_idx = cd_span.index(idx)
        end_idx = len(cd_span) - cd_span[::-1].index(idx)

        for j in range(idx + 1, len(s_slist)):
            jdx = s_slist[j]
            begin_jdx = cd_span.index(jdx)
            end_jdx = len(cd_span) - cd_span[::-1].index(jdx)
            for w_i in range(begin_idx,end_idx):
                for w_j in range(begin_jdx,end_jdx):
                    st_adj[w_i][w_j] = 0
                    st_adj[w_j][w_i] = 0
    return st_adj


def form_aspect_related_spans(aspect_node_idx, spans, mapnode, node2layerid, path_dict,select_N = ['ROOT','TOP','S','NP','VP'], special_token = '[N]'):
    aspect2root_path = path_dict[aspect_node_idx]
    span_indications = []
    spans_range = []
    
    for idx,f in enumerate(aspect2root_path[:-1]):
        if mapnode[f][:-len(special_token)] in select_N:
            span_idx = node2layerid[f]
            span_temp = spans[span_idx]

            if len(spans_range) == 0 or span_temp != spans_range[-1]:
                spans_range.append(span_temp)
                span_indications.append(mapnode[f][:-len(special_token)])
        
    return spans_range, span_indications





def select_func(spans, max_num_spans, length):
    if len(spans) <= max_num_spans:
        lacd_span = spans[-1] if len(spans) > 0 else [0] * length
        select_spans = spans + [lacd_span] * (max_num_spans - len(spans))

    else:
        if max_num_spans == 1:
            select_spans = spans[0] if len(spans) > 0 else [0] * length
        else:
            gap = len(spans)  // (max_num_spans-1)
            select_spans = [ spans[gap * i] for i in range(max_num_spans-1)] + [spans[-1]]

    return select_spans