import os
import pdb
import torch
import argparse
import tensorboardX
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import sparse
from torch.autograd import Variable

import utils
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader
from enc_dec.geo_mosgraphflow_analysis_decoder import TSGNNDecoder

# PARSE ARGUMENTS FROM COMMAND LINE
def arg_parse():
    parser = argparse.ArgumentParser(description='COEMBED ARGUMENTS.')
    # ADD FOLLOWING ARGUMENTS
    parser.add_argument('--cuda', dest = 'cuda',
                help = 'CUDA.')
    parser.add_argument('--parallel', dest = 'parallel',
                help = 'Parrallel Computing')
    parser.add_argument('--GPU IDs', dest = 'gpu_ids',
                help = 'GPU IDs')
    parser.add_argument('--add-self', dest = 'adj_self',
                help = 'Graph convolution add nodes themselves.')
    parser.add_argument('--model', dest = 'model',
                help = 'Model load.')
    parser.add_argument('--lr', dest = 'lr', type = float,
                help = 'Learning rate.')
    parser.add_argument('--batch-size', dest = 'batch_size', type = int,
                help = 'Batch size.')
    parser.add_argument('--num_workers', dest = 'num_workers', type = int,
                help = 'Number of workers to load data.')
    parser.add_argument('--epochs', dest = 'num_epochs', type = int,
                help = 'Number of epochs to train.')
    parser.add_argument('--input-dim', dest = 'input_dim', type = int,
                help = 'Input feature dimension')
    parser.add_argument('--hidden-dim', dest = 'hidden_dim', type = int,
                help = 'Hidden dimension')
    parser.add_argument('--output-dim', dest = 'output_dim', type = int,
                help = 'Output dimension')
    parser.add_argument('--num-gc-layers', dest = 'num_gc_layers', type = int,
                help = 'Number of graph convolution layers before each pooling')
    parser.add_argument('--dropout', dest = 'dropout', type = float,
                help = 'Dropout rate.')

    # SET DEFAULT INPUT ARGUMENT
    parser.set_defaults(cuda = '0',
                        parallel = False,
                        add_self = '0', # 'add'
                        model = '0', # 'load'
                        lr = 0.001,
                        clip = 2.0,
                        batch_size = 1,
                        num_workers = 1,
                        num_epochs = 100,
                        input_dim = 10,
                        hidden_dim = 10,
                        output_dim = 30,
                        decoder_dim = 150,
                        dropout = 0.1)
    return parser.parse_args()

def build_geotsgnn_model(args, device, graph_output_folder,num_class, fold_n):
    print('--- BUILDING UP TSGNN MODEL ... ---')
    # GET PARAMETERS
    # [num_gene, num_drug, (adj)node_num]
    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    node_num = len(gene_name_list)
    # [num_gene_edge, num_drug_edge]
    form_data_path = './' + graph_output_folder + '/form_data'
    edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long()
    num_edge = edge_index.shape[1]
    # import pdb; pdb.set_trace()
    # BUILD UP MODEL
    model = TSGNNDecoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim, decoder_dim=args.decoder_dim,
                node_num=node_num, num_edge=num_edge, device=device,graph_output_folder=graph_output_folder,num_class=num_class, fold_n=fold_n)
    model = model.to(device)
    return model


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    

def analysis_geotsgnn_model(dataset_loader, batch_random_final_dl_input_df, analysis_save_path, model, device, args,graph_output_folder):
    batch_loss = 0
    for batch_idx, data in enumerate(dataset_loader):
        x = Variable(data.x, requires_grad=False).to(device)
        edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
        #assert label.min() >= 0, f"Label min is less than 0: {label.min().item()}"
        #assert label.max() < num_class, f"Label max is out of range: {label.max().item()}"
        #adj = Variable(adj.float(), requires_grad=False).to(device)
        # THIS WILL USE METHOD [def forward()] TO MAKE PREDICTION
        output, ypred = model(x, edge_index, internal_edge_index, graph_output_folder, batch_random_final_dl_input_df, analysis_save_path)
        # batch_ypred = ypred
        # print('Batch prediction shape:', batch_ypred.shape)
        loss = model.loss(output, label)
        batch_loss += loss.item()
    return model, batch_loss


def analysis_geotsgnn(args, fold_n, model, analysis_save_path, device, graph_output_folder):
    print('-------------------------- ANALYSIS START --------------------------')
    print('-------------------------- ANALYSIS START --------------------------')
    print('-------------------------- ANALYSIS START --------------------------')
    print('-------------------------- ANALYSIS START --------------------------')
    print('-------------------------- ANALYSIS START --------------------------')
    # ANALYSIS MODEL ON WHOLE DATASET
    form_data_path = './' + graph_output_folder + '/form_data'
    xAll = np.load(form_data_path + '/xAll.npy')
    yAll = np.load(form_data_path + '/yAll.npy')
    # xTe = np.load(form_data_path + '/xTe' + str(fold_n) + '.npy')
    # yTe = np.load(form_data_path + '/yTe' + str(fold_n) + '.npy')
    random_final_dl_input_df = pd.read_csv('./' + graph_output_folder + '/random-survival-label.csv')
    # READ [adj, edge_index] FILES 
    #adj = sparse_mx_to_torch_sparse_tensor(sparse.load_npz(form_data_path + '/adj_sparse.npz'))
    edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long() 
    internal_edge_index = torch.from_numpy(np.load(form_data_path + '/internal_edge_index.npy') ).long()

    dl_input_num = xAll.shape[0]
    batch_size = args.batch_size
    # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
    # [num_feature, num_gene, num_drug]
    num_feature = 10

    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    num_node = final_annotation_gene_df.shape[0]
    # [num_cellline]
    survival_label_list = sorted(list(set(random_final_dl_input_df['individualID'])))
    survival_label_num = [x for x in range(1, len(survival_label_list)+1)]
    survival_label_map_df = pd.DataFrame({'individualID': survival_label_list, 'individualID_Num': survival_label_num})
    survival_label_map_df.to_csv('./' + graph_output_folder + '/survival_label_map_dict.csv', index=False, header=True)
    batch_included_survival_label_list = []
    # RUN ANALYSIS MODEL
    model.eval()
    # all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, dl_input_num, batch_size):
        if (index + batch_size) < dl_input_num:
            upper_index = index + batch_size
        else:
            upper_index = dl_input_num
        geo_datalist = read_batch(index, upper_index, xAll, yAll, num_feature, num_node, edge_index, internal_edge_index, graph_output_folder)
        dataset_loader, node_num, feature_dim = GeoGraphLoader.load_graph(geo_datalist, prog_args)
        batch_random_final_dl_input_df = random_final_dl_input_df.iloc[index : upper_index]
        print('ANALYZE MODEL...')
        # import pdb; pdb.set_trace()
        model, batch_loss = analysis_geotsgnn_model(dataset_loader, 
                                            batch_random_final_dl_input_df, analysis_save_path, model, device, args,graph_output_folder)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        # # PRESERVE PREDICTION OF BATCH TEST DATA
        # batch_ypred = (Variable(batch_ypred).data).cpu().numpy()
        # all_ypred = np.vstack((all_ypred, batch_ypred))
        # all_ypred = np.concatenate((all_ypred, batch_ypred.reshape(-1, 1)), axis=0)
        # TIME TO STOP SINCE ALL [cell line] WERE INCLUDED
        tmp_batch_survival_label_list = sorted(list(set(batch_random_final_dl_input_df['individualID'])))
        batch_included_survival_label_list += tmp_batch_survival_label_list
        batch_included_survival_label_list = sorted(list(set(batch_included_survival_label_list)))
        # import pdb; pdb.set_trace()
        if batch_included_survival_label_list == survival_label_list:
            print(len(batch_included_survival_label_list))
            print(batch_included_survival_label_list)
            break


if __name__ == "__main__":
    # Parse argument from terminal or default parameters
    prog_args = arg_parse()

    # Check and allocate resources
    device, prog_args.gpu_ids = utils.get_available_devices()
    # Manual set
    device = torch.device('cuda:0') 
    torch.cuda.set_device(device)
    print('MAIN DEVICE: ', device)
    # Single gpu
    prog_args.gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Dataset Selection
    dataset = 'ROSMAP'
    graph_output_folder = dataset + '-graph-data'

    k = 5
    for fold_n in np.arange(1, k + 1):
        os.makedirs('./' + dataset + '-analysis/fold_' + str(fold_n), exist_ok=True)
        graph_output_folder = dataset + '-graph-data'
        yTr = np.load('./' + graph_output_folder + '/form_data/yTr' + str(fold_n) + '.npy')
        unique_numbers, occurrences = np.unique(yTr, return_counts=True)
        num_class = len(unique_numbers)
        print("num:" ,num_class)

        model = build_geotsgnn_model(prog_args, device, graph_output_folder,num_class, fold_n)
        ### TEST THE MODEL
        analysis_load_path = './' + dataset + '-result/mosgraphflow/fold_' + str(fold_n) + '/best_train_model.pt'
        analysis_save_path = './' + dataset + '-analysis/fold_' + str(fold_n)
        model.load_state_dict(torch.load(analysis_load_path, map_location=device))
        analysis_geotsgnn(prog_args, fold_n, model, analysis_save_path, device, graph_output_folder)
