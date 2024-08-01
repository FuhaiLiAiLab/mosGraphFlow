import os
import pdb
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from matplotlib.pyplot import figure
from sklearn.preprocessing import normalize
from matplotlib.ticker import PercentFormatter


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def generate_edge_type(row, node_mapping):
    from_node = row['From']
    to_node = row['To']
    from_type = node_mapping.get(from_node, 'Unknown')
    to_type = node_mapping.get(to_node, 'Unknown')
    return f"{from_type}-{to_type}"

class PathNetAnalyse():
    def __init__(self):
        pass


    # TODO  kegg_gene_num_dict.csv
    def prepare_network(self, graph_output_folder, dataset):
        ### GET [node_num_dict] FOR WHOLE NET NODES
        kegg_gene_num_dict_df = pd.read_csv('./' + graph_output_folder + '/map-all-gene.csv')
        kegg_gene_num_dict_df = kegg_gene_num_dict_df.rename(columns={'Gene_name': 'node_name', 'Gene_num': 'node_num', 'NodeType': 'node_type'})
        node_num_dict_df = kegg_gene_num_dict_df
        node_num_dict_df = node_num_dict_df[['node_num', 'node_name', 'node_type']]
        node_num_dict_df.to_csv('./' + dataset + '-analysis' + '/node_num_dict.csv', index=False, header=True)

    def sp_kth_hop_att_network(self, fold_n=1, sp=1, khop_sum=3, survival_label = '1',graph_output_folder = 'ROSMAP-graph-data', dataset='ROSMAP'):

        survival_label_map_df = pd.read_csv('./' + graph_output_folder + '/survival_label_map_dict.csv')
        survival_label_map_dict = dict(zip(survival_label_map_df.individualID, survival_label_map_df.individualID_Num))
        survival_label_num = survival_label_map_dict[survival_label]

        sp_attention_file = './' + dataset + '-analysis' + '/fold_' + str(fold_n) + '/sp' + str(sp) + '/survival' + str(survival_label_num)
        sp_survival_label_df = pd.read_csv(sp_attention_file + '.csv')
        
        # FILTER OUT ROWS WITH [mask==0]
        sp_survival_label_df.drop(sp_survival_label_df[sp_survival_label_df['Mask'] == 0.0].index, inplace=True)
        sp_survival_label_df = sp_survival_label_df.reset_index(drop=True)
        # ADD [survival line] & [signaling_path]
        kegg_sp_map_df = pd.read_csv('./' + graph_output_folder + '/kegg_sp_map.csv')
        kegg_sp_map_dict = dict(zip(kegg_sp_map_df.SpNotation, kegg_sp_map_df.SignalingPath))

        SpNotation = 'sp' + str(sp)
        SignalingPath = kegg_sp_map_dict[SpNotation]
        SpNotation_list = [SpNotation] * (sp_survival_label_df.shape[0])
        SignalingPath_list = [SignalingPath] * (sp_survival_label_df.shape[0])
        # REPLACE [sub_idx] WITH [node_idx]
        sp_map_file_path = './' + graph_output_folder + '/form_data/sp' + str(sp) + '_gene_map.csv'
        sp_map_df = pd.read_csv(sp_map_file_path)
        sp_map_dict = dict(zip(sp_map_df.Sub_idx, sp_map_df.Node_idx))
        sp_survival_label_df = sp_survival_label_df.replace({'From': sp_map_dict, 'To': sp_map_dict})
        sp_survival_label_df = sp_survival_label_df[['From', 'To', 'Attention', 'Hop']]
        sp_survival_label_df['SignalingPath'] = SignalingPath_list
        sp_survival_label_df['SpNotation'] = SpNotation_list
        sp_survival_label_df['individualID'] = [survival_label] * (sp_survival_label_df.shape[0])
        # REPLACE ORIGINAL FILE WITH FILTERED ONE
        sp_survival_label_df.to_csv(sp_attention_file + '_filtered.csv', index=False, header=True)
        # SEPARATE SEVERAL [cell line]
        kth_hop_sp_df_list = []
        for khop_num in range(1, khop_sum + 1):
            khop_str = 'hop' + str(khop_num)
            kth_hop_sp_df = sp_survival_label_df[sp_survival_label_df['Hop']==khop_str]
            kth_hop_sp_df.to_csv(sp_attention_file + '_' + khop_str + '.csv', index=False, header=True)
            kth_hop_sp_df_list.append(kth_hop_sp_df)
        return sp_survival_label_df, kth_hop_sp_df_list

    def organize_survival_label_specific_network(self, fold_n, graph_output_folder, dataset):
        # SURVIVAL LABEL LIST
        survival_label_map_df = pd.read_csv('./' + graph_output_folder + '/survival_label_map_dict.csv')
        survival_label_list = list(survival_label_map_df['individualID'])
        survival_label_map_dict = dict(zip(survival_label_map_df.individualID, survival_label_map_df.individualID_Num))
        # SIGNALING PATHWAY LIST
        kegg_sp_map_df = pd.read_csv('./' + graph_output_folder + '/kegg_sp_map.csv')
        kegg_sp_num = kegg_sp_map_df.shape[0]
        # COLLECT ALL SURVIVAL LABEL SPECIFIC ATTENTION DATA FRAME
        for survival_label in survival_label_list:
            survival_specific_combined_sp_khop_df_list = []
            survival_label_num = survival_label_map_dict[survival_label]
            for sp in range(1, kegg_sp_num + 1):
                sp_survival_label_df, kth_hop_sp_df_list = PathNetAnalyse().sp_kth_hop_att_network(fold_n=fold_n, sp=sp, khop_sum=3, survival_label = survival_label)
                survival_specific_combined_sp_khop_df_list.append(sp_survival_label_df)
            comtemp_survival_specific_combined_sp_khop_df = pd.concat(survival_specific_combined_sp_khop_df_list)
            comtemp_survival_specific_combined_sp_khop_df.to_csv('./' + dataset + '-analysis' + '/fold_' + str(fold_n) + '/survival' + str(survival_label_num) + '.csv', index=False, header=True)
            comtemp_survival_specific_combined_sp_khop_df.to_csv('./' + dataset + '-analysis' + '/fold_' + str(fold_n) + '_survival/survival' + str(survival_label_num) + '.csv', index=False, header=True)
            

class AverageFoldPath():
    def __init__(self):
        pass

    def average_fold_edge(self, dataset):
        ### [fold_0 is averaged path weight]
        if os.path.exists('./' + dataset + '-analysis' + '/avg_survival') == False:
            os.mkdir('./' + dataset + '-analysis' + '/avg_survival')

        if os.path.exists('./' + dataset + '-analysis' + '/avg') == False:
            os.mkdir('./' + dataset + '-analysis' + '/avg')
        # survival LINE LIST
        survival_label_map_df = pd.read_csv('./' + graph_output_folder + '/survival_label_map_dict.csv')
        survival_label_num = survival_label_map_df.shape[0]

        node_dict_file = './ROSMAP-analysis/node_num_dict.csv'
        node_dict = pd.read_csv(node_dict_file)
        node_mapping = dict(zip(node_dict['node_num'], node_dict['node_type']))

        for survival_num in range(1, survival_label_num + 1):
            fold_survival_edge_df_list = []
            for fold_num in range(1, 6):
                fold_survival_path = './' + dataset + '-analysis' + '/fold_' + str(fold_num) + '_survival/survival' + str(survival_num) +'.csv'
                fold_survival_edge_df = pd.read_csv(fold_survival_path)

                # Only keep rows where Hop is 'hop1'
                fold_survival_edge_df = fold_survival_edge_df[fold_survival_edge_df['Hop'] == 'hop1']

                # add edge type
                fold_survival_edge_df['EdgeType'] = fold_survival_edge_df.apply(generate_edge_type, axis=1, args=(node_mapping,))
                fold_survival_edge_df['Attention'] = fold_survival_edge_df['Attention']

                fold_survival_edge_df_list.append(fold_survival_edge_df)
            fold_survival_group_df = pd.concat(fold_survival_edge_df_list)
            fold_survival_group_df = fold_survival_group_df.apply(pd.to_numeric, errors='coerce')
            fold_survival_group_df = fold_survival_group_df.groupby(level=0).mean()

            fold_survival_average_df = fold_survival_edge_df
                        # convert column ['From', 'To'] to int
            fold_survival_group_df[['From', 'To']] = fold_survival_group_df[['From', 'To']].astype(int)

            fold_survival_average_df.to_csv('./' + dataset + '-analysis' + '/avg/survival' + str(survival_num) +'.csv', index=False, header=True)
        

### DATASET SELECTION
dataset = 'ROSMAP'
graph_output_folder = dataset + '-graph-data'


for fold_n in range(1, 6):
    PathNetAnalyse().prepare_network(graph_output_folder, dataset)
    if os.path.exists('./' + dataset + '-analysis' + '/fold_' + str(fold_n) +'_survival') == False:
        os.mkdir('./' + dataset + '-analysis' + '/fold_' + str(fold_n) +'_survival')

    PathNetAnalyse().organize_survival_label_specific_network(fold_n, graph_output_folder, dataset)

AverageFoldPath().average_fold_edge(dataset)