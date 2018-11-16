folder_ratio_map = {
    'pos':   1,
    'neg':   1,                
            
}

bin_class_map_dict = {
    'pos':    1,
    'neg':    0,       
}
# all negtive as 0, positive can have multiple classes, 1 is very bad positive, 2 stands for negtive.
# 0 is negtive

multi_class_map_dict = {
    'pos':    1,
    'neg':    0,                
      
}


class_reverse_map = {}
for k, v in multi_class_map_dict.items():
    class_reverse_map[v] = k

folder_map_dict = {}
for idx, (k, v ) in enumerate(folder_ratio_map.items()):
    folder_map_dict[k] = idx

folder_reverse_map = {}
for k, v in folder_map_dict.items():
    folder_reverse_map[v] = k

def get_pos_neg(multi_label_list):
    # multi_label_list
    ret_label_list = []
    if multi_label_list is not None: 
        for this_label in multi_label_list:
            pos_neg_label = bin_class_map_dict[ class_reverse_map[this_label] ] 
            ret_label_list.append( pos_neg_label )
        return ret_label_list
    else:
        return None
