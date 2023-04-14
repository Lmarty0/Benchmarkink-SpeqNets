import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from graph_tool.all import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
from sklearn.metrics import f1_score
import time
import matplotlib.pyplot as plt
import math
import torch.nn as nn
from torch_geometric.datasets import GEDDataset
from torch_scatter import scatter
from torch_geometric.transforms import OneHotDegree
from torch.nn import MSELoss
from torch_geometric.transforms import NormalizeFeatures
from data_utils_ged import load_ged_dataset
import random

class Data_List(object) : 
    def __init__(self, graph, dataset) :
        
        self.dataset = dataset
        self.FINAL_DIM = max([data.num_nodes for data in self.dataset])
        self.data = graph
        self.data = self.compute_degree_vector(self.data)
        self.x = self.data.x.cpu().detach().numpy()
        self.edge_index = self.data.edge_index.cpu().detach().numpy()
        # Create graph for easier processing.
        g = Graph(directed=False)
        num_nodes = self.x.shape[0]

        node_features = {}
        for i in range(num_nodes):
            v = g.add_vertex()
            node_features[v] = self.x[i]

        rows = list(self.edge_index[0])
        cols = list(self.edge_index[1])

        for ind, (i, j) in enumerate(zip(rows, cols)):
            g.add_edge(i, j, add_missing=False)

        tuple_graph = Graph(directed=False)
        type = {}

        tuple_to_nodes = {}
        nodes_to_tuple = {}
        for v in g.vertices():
            for w in v.all_neighbors():
                n = tuple_graph.add_vertex()
                tuple_to_nodes[n] = (v, w)
                nodes_to_tuple[(v, w)] = n

                type[n] = np.concatenate(
                    [node_features[v], node_features[w], np.array([1, 0])], axis=-1)

            n = tuple_graph.add_vertex()
            tuple_to_nodes[n] = (v, v)
            tuple_to_nodes[(v, v)] = n
            type[n] = np.concatenate([node_features[v], node_features[v], np.array([0, 1])], axis=-1)

        matrix_1 = []
        matrix_2 = []
        node_features = []

        index_1 = []
        index_2 = []

        for t in tuple_graph.vertices():
            # Get underlying nodes.
            v, w = tuple_to_nodes[t]

            node_features.append(type[t])
            index_1.append(int(v))
            index_2.append(int(w))

            # 1 neighbors.
            for n in v.out_neighbors():
                if (n, w) in nodes_to_tuple:
                    s = nodes_to_tuple[(n, w)]
                    matrix_1.append([int(t), int(s)])

            # 2 neighbors.
            for n in w.out_neighbors():
                if (v, n) in nodes_to_tuple:
                    s = nodes_to_tuple[(v, n)]
                    matrix_2.append([int(t), int(s)])

        self.data_list = []

        data_new = Data()

        edge_index_1 = torch.tensor(matrix_1).t().contiguous()
        edge_index_2 = torch.tensor(matrix_2).t().contiguous()

        self.edge_index_1 = edge_index_1
        self.edge_index_2 = edge_index_2

        self.x = torch.from_numpy(np.array(node_features)).to(torch.float)
        self.index_1 = torch.from_numpy(np.array(index_1)).to(torch.int64)
        self.index_2 = torch.from_numpy(np.array(index_2)).to(torch.int64)

        self.y = self.data.y
    
    def compute_degree_vector(self, data):
    # Create a tensor of ones with the same size as the edge index
        num_nodes =  self.FINAL_DIM
        deg_onehot = OneHotDegree(num_nodes, cat = False)(data)
        return deg_onehot

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        dim = 256
        self.conv_1_1 = GCNConv(input_dim, dim)
        self.conv_1_2 = GCNConv(input_dim, dim)

        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.conv_2_1 = GCNConv(dim, dim)
        self.conv_2_2 = GCNConv(dim, dim)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.mlp = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, 7))
        self.dist_mlp = MLP(7,16,1)
        self.apply(self._init_weights)


    # At initialization of the weights according to a normal distribution
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, data1, data2):
        
        #Get the embedding for the first graph
        x, edge_index_1, edge_index_2 = data1.x, data1.edge_index_1, data1.edge_index_2
        index_1, index_2 = data1.index_1, data1.index_2
        x_1 = F.relu(self.conv_1_1(x, edge_index_1))
        x_2 = F.relu(self.conv_1_2(x, edge_index_2))
        x = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv_2_1(x, edge_index_1))
        x_2 = F.relu(self.conv_2_2(x, edge_index_2))
        x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        index_1 = index_1.to(torch.int64)
        index_2 = index_2.to(torch.int64)
        x_1 = scatter(x, index_1, dim=0, reduce="mean")
        x_2 = scatter(x, index_2, dim=0, reduce="mean")
        embed_1 = self.mlp(torch.cat([x_1, x_2], dim=1))
        embed_1 = embed_1.sum(dim=0)


        #Get the Embedding for the second graph 
        x, edge_index_1, edge_index_2 = data2.x, data2.edge_index_1, data2.edge_index_2
        index_1, index_2 = data2.index_1, data2.index_2
        x_1 = F.relu(self.conv_1_1(x, edge_index_1))
        x_2 = F.relu(self.conv_1_2(x, edge_index_2))
        x = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv_2_1(x, edge_index_1))
        x_2 = F.relu(self.conv_2_2(x, edge_index_2))
        x = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        index_1 = index_1.to(torch.int64)
        index_2 = index_2.to(torch.int64)
        x_1 = scatter(x, index_1, dim=0, reduce="mean")
        x_2 = scatter(x, index_2, dim=0, reduce="mean")
        embed_2 = self.mlp(torch.cat([x_1, x_2], dim=1))
        embed_2 = embed_2.sum(dim=0)

        dist = self.dist_mlp(embed_1+embed_2)


        return dist

class SpeqNet_for_GSL(object) : 
    def __init__(self, dataset) : 
        self.name= dataset

        self.idx_training = self.get_subsets(self.name,1)
        if self.name == 'LINUX' : 
            self.idx_val = [x for x in range(601, 800)]
        elif self.name == 'AIDS700nef' : 
            self.idx_val = [x for x in range(421, 560)]

        self.path = osp.join(osp.dirname(osp.realpath(__file__)),'data', self.name)
        self.dataset = GEDDataset(self.path, self.name, train=True, transform=NormalizeFeatures())
        self.dataset2 = GEDDataset(self.path, self.name, train=True, transform=NormalizeFeatures())
        self.test_dataset = GEDDataset(self.path, self.name, train=False,transform=NormalizeFeatures() )
        self.val_dataset = self.dataset[self.idx_val]

        FINAL_DIM = max([data.num_nodes for data in self.dataset])
        INPUT_DIM  = Data_List(self.dataset[0], self.dataset).x.shape[1]
        self.model = Net(input_dim= INPUT_DIM)
   
        self.loss_funct = MSELoss(reduce=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=5e-3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_subsets(self, name, ratio):
        """
        param `name`: name of GED Dataset. str. can be AIDS700nef or LINUX
        param `ratio`: float. can be in [0, 1].
        """
        if not (ratio >= 0 and ratio <= 1):
            raise ValueError(f"Ratio needs to be in [0, 1]. Found {ratio}.")
        dataset = load_ged_dataset(name)
        total = len(dataset)
        validation = {'AIDS700nef': 140, 'LINUX': 200}
        total = total - validation[name] # graphs from end will be in validation set
        perm = np.random.RandomState(seed=0).permutation(total)
        # A fixed seed=0 ensures larger sets are supersets of smaller sets
        subset_size = int(np.ceil(ratio * total))
        gids = perm[:subset_size]
        return gids

    def get_target_vector(self, dataset, idx1, idx2) : 
        T=torch.zeros((len(idx1), 1))
        for i in range(len(idx1)) :
            dist = dataset.ged[self.dataset[idx1[i]].i, self.dataset[idx2[i]].i]
            T[i][0] = dist
        return T
    

    def train(self, idx1, idx2) : 
        self.model.train() 
        self.optimizer.zero_grad()
        
        all_dist = []
        for i in range(len(idx1)) :
            data1 = Data_List(self.dataset[idx1[i]], self.dataset)
            data2 = Data_List(self.dataset[idx2[i]], self.dataset)
            dist = self.model(data1,data2)
            all_dist.append(dist)
        pred = torch.cat(all_dist)
        target = self.get_target_vector(self.dataset, idx1, idx2)
        loss = self.loss_funct(pred, target)
        loss = loss.squeeze(dim=1).sum()
        loss.backward()
        target = target.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        train_rmse = math.sqrt(np.square(np.subtract(target, pred)).mean())
        self.optimizer.step()
        return loss, train_rmse

    def get_F1_topK_score(self, pred, true,K) : 
        ind_p= np.argpartition(pred, -K)[-K:]
        ind_t = np.argpartition(true, -K)[-K:]
        ind_p = ind_p[np.argsort(pred[ind_p])]
        ind_t = ind_t[np.argsort(true[ind_t])]
        f1_scr = f1_score(ind_t, ind_p, average = 'macro')
        return f1_scr


    @torch.no_grad()
    def test(self, batch_size= 10) : 
        self.model.eval()
        idx_testing = [i for i in range(0,len(self.test_dataset))]

        
        #Validation set
        val_rmse_epoch = 0
        val_loss_epoch = 0

        impl = 0
        self.idx_batch_val = [self.idx_val[i*10:(i+1)*10] for i in range(len(self.idx_val)//10)]
        for idx_v in self.idx_batch_val : 
            for idx1 in self.idx_batch1 : 
                if len(idx_v) == batch_size and len(idx1) == batch_size : 
                    impl +=1
                    all_dist = []
                    for i in range(len(idx1)) :
                        data1 = Data_List(self.dataset[idx1[i]], self.dataset)
                        data2 = Data_List(self.dataset[idx_v[i]], self.dataset)
                        dist = self.model(data1,data2)
                        all_dist.append(dist)

                    pred = torch.cat(all_dist)
                    true = self.get_target_vector(self.dataset, idx_v, idx1)
                    true = torch.squeeze(true,1)
                    loss_val = self.loss_funct(pred, true)
                    loss_val = loss_val.sum()
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    rmse_val = math.sqrt(np.square(np.subtract(true, pred)).mean())
                    val_rmse_epoch += rmse_val
                    val_loss_epoch += loss_val.item()
        
        if impl != 0 :
            val_rmse_epoch = val_rmse_epoch /impl
            val_loss_epoch = val_loss_epoch /impl
        else :
            val_rmse_epoch = 0
            val_loss_epoch = 0

        #Training set
        test_rmse_epoch = 0
        test_f1_k_epoch = 0
        impl = 0

        self.idx_batch_test = [idx_testing[i*10:(i+1)*10] for i in range(len(idx_testing)//10)]
        for idx_t in self.idx_batch_test : 
            for idx1 in self.idx_batch1 :
                if len(idx_t)== batch_size and len(idx1)== batch_size :  
                    impl +=1
                    all_dist = []
                    for i in range(len(idx1)) :
                        data1 = Data_List(self.dataset[idx1[i]], self.dataset)
                        data2 = Data_List(self.test_dataset[idx_t[i]], self.test_dataset)
                        dist = self.model(data1,data2)
                        all_dist.append(dist)
                    pred = torch.cat(all_dist)
                    true = self.get_target_vector(self.dataset, idx_t, idx1).detach().cpu().numpy()
                    #pred = np.squeeze(pred, axis=1)
                    true = np.squeeze(true, axis= 1)
                    f1_scr = self.get_F1_topK_score(pred, true, K =5)
                    rmse_test = math.sqrt(np.square(np.subtract(true, pred)).mean())
                    test_rmse_epoch += rmse_test
                    test_f1_k_epoch += f1_scr
        if impl != 0 : 
            test_rmse_epoch = test_rmse_epoch/impl
            test_f1_k_epoch = test_f1_k_epoch/impl
        else :
            test_rmse_epoch = 0
            test_f1_k_epoch = 0

        return val_rmse_epoch, val_loss_epoch, test_rmse_epoch, test_f1_k_epoch

    def write_report(self):
        final_time = time.time() - self.starting_time
        with open('Report Training - GCN - '+ self.name +'- GSL.txt', 'w') as f:
            f.write('Dataset  : ' + self.name +'\n')
            f.write('Number of epochs of training : ' + str(self.epoch)+'\n')
            f.write('Time of training  : ' + str(final_time) +'seconds \n')    
            f.write('Final RMSE Score : ' + str(self.RMSE_Test[-1])+'\n')
            f.write('Final F1-Top K Score : '+ str(self.F1_K_Test[-1]))


    def full_training(self, volume_training = False, batch_size = 10) : 

        if volume_training == True : 
            self.current_idx = self.idx_training2            

        else : 
            self.current_idx = self.idx_training
            
        print('Volume of training data ', len(self.current_idx))

        self.idx_batch1 = [self.current_idx[i*10:(i+1)*10] for i in range(len(self.current_idx)//10)]
        random.shuffle(self.current_idx)
        self.idx_batch2 = [self.current_idx[i*10:(i+1)*10] for i in range(len(self.current_idx)//10)]
        
        self.Epochs = []
        self.Train_Loss = []
        self.Validation_Loss =[]
        self.RMSE_Test =[]
        self.F1_K_Test = []
        self.Training_Time=[]

        #Early Stopping
        patience = 30
        trigger_value = 0 

        self.starting_time = time.time()
        self.epoch = 0

        lowest_val_loss  = float('inf')

        for idx1 in self.idx_batch1 : 
            for idx2 in self.idx_batch2 :
                if len(idx1) == batch_size and len(idx2)==batch_size : 
                    self.epoch += 1
                    
                    train_loss, train_rmse = self.train(idx1, idx2)

                    if self.epoch % 300  == 0:
                        
                        curr_time = time.time() - self.starting_time
                        val_rmse_epoch, val_loss_epoch, test_rmse_epoch, test_f1_k_epoch= self.test()
                        self.Epochs.append(self.epoch)
                        self.Train_Loss.append(train_loss.item())
                        self.Validation_Loss.append(val_loss_epoch)
                        self.RMSE_Test.append(test_rmse_epoch)
                        self.F1_K_Test.append(test_f1_k_epoch)
                        self.Training_Time.append(curr_time)

                        if val_loss_epoch < lowest_val_loss : 
                            lowest_val_loss = val_loss_epoch
                            trigger_value = 0

                        elif val_loss_epoch >= lowest_val_loss : 
                            trigger_value +=1
                            if trigger_value >= patience : 

                                print('Early Stopping at Epoch', self.epoch)
                                print(f'Epoch: {self.epoch:03d} '
                                        f'Val: {val_rmse_epoch:.4f}, Test: {test_rmse_epoch:.4f}')
                                
                                if volume_training == True :
                                    return self.RMSE_Test, self.Train_Loss, self.Validation_Loss, self.Epochs, self.F1_K_Test
                                else :
                                    self.write_report()
                                    return self.RMSE_Test, self.Train_Loss, self.Validation_Loss, self.Epochs, self.F1_K_Test                                
                            
                                    
                    

                    print('Epoch : ', self.epoch, 'out of ', len(self.idx_batch1)*len(self.idx_batch2))
                    print('Train Loss : ', train_loss.item())
                    #print('Train RMSE : ', test_rmse_epoch)

        return self.RMSE_Test, self.Train_Loss, self.Validation_Loss, self.Epochs, self.F1_K_Test

    def get_training_plots(self) : 

        #Plotting
        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(30)
        fig.set_figwidth(30)
        fig.suptitle('Training performances of SpeqNet2_1 - on '+ self.name+' - With Early Stopping')
        axs[0,0].plot(self.Epochs, self.RMSE_Test)
        axs[0,0].set_title('Performance vs Epochs')
        axs[0,0].set_xlabel('Epoch')
        axs[0,1].plot(self.Epochs, self.Train_Loss, label='Training Loss')
        axs[0,1].plot(self.Epochs, self.Validation_Loss, label='Validation Loss')
        axs[0,1].legend()
        axs[0,1].set_title('Training and Validation Loss vs Epochs')
        axs[0,1].set_xlabel('Epoch')
        axs[1,0]. plot(self.Training_Time, self.RMSE_Test)
        axs[1,0].set_title('Performance vs Training Time')
        axs[1,0].set_xlabel('Training Time (s)')
        axs[1,1].plot(self.Training_Time, self.Train_Loss, label='Training Loss')
        axs[1,1].plot(self.Training_Time, self.Validation_Loss, label="Validation Loss")
        axs[1,1].set_title('Training and Validation Loss vs Training Time')
        axs[1,1].set_xlabel('Training Time (s)')
        axs[1,1].legend()
        fig.savefig('Benchmark SpeqNet2_1 -' + self.name+ '- Graph Similarity Learning.png')
        plt.show()

    def volume_training(self) : 
        self.Vol_RMSE, self.Vol_F1, self.Vol_TL, self.Vol_VL, self.Vol_epochs = {}, {},{},{}, {}
        self.Vol_data = []
        for ratio in range(1,5) :
            ratio = ratio/10
            #Initalize the weights of the model
            self.model.apply(self.model._init_weights)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=5e-3)

            self.idx_training2 = self.get_subsets(self.name,ratio)
            rmse_list, train_loss_list, val_loss_list, epochs_list, f1_k_list = self.full_training(volume_training=True)
            vol_data = len(self.idx_training)
            self.Vol_data.append(vol_data)
            self.Vol_F1[vol_data] = f1_k_list
            self.Vol_RMSE[vol_data] = rmse_list
            self.Vol_TL[vol_data] = train_loss_list
            self.Vol_VL[vol_data] = val_loss_list
            self.Vol_epochs[vol_data] = epochs_list

        # Plotting the result

        fig, axs = plt.subplots(1, 4)
        fig.set_figheight(15)
        fig.set_figwidth(30)
        fig.suptitle('Training performances of SpeqNet2_1 - on '+ self.name+' - With Early Stopping for Graph Similarity Learning')



        for vol in self.Vol_TL : 
            axs[0].plot(self.Vol_epochs.get(vol), self.Vol_RMSE.get(vol), label=str(vol)+' graphs for training')
            axs[1].plot(self.Vol_epochs.get(vol), self.Vol_TL[vol], label=str(vol)+' graphs for training')
            axs[2].plot(self.Vol_epochs[vol], self.Vol_VL[vol], label=str(vol)+' graphs for training')
            axs[3].plot(self.Vol_epochs.get(vol), self.Vol_F1.get(vol), label=str(vol)+' graphs for training')

        axs[0].set_title('RMSE vs Epochs')
        axs[0].set_xlabel('Epoch')
        axs[1].set_title('Training Loss vs Epochs')
        axs[1].set_xlabel('Epochs')
        axs[2].set_title('Validation Loss vs Epochs')
        axs[2].set_xlabel('Epochs')
        axs[3].set_title('F1-Top k vs Epochs')
        axs[3].set_xlabel('Epoch')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[3].legend()
        fig.savefig('Benchmark - GCN - '+ self.name + ' - Graph Similarity Learning - Volume training.png')
        print('Plot saved')
