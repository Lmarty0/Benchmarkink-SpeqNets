import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from graph_tool.all import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
from sklearn.metrics import f1_score
import time
import matplotlib.pyplot as plt
import math


class PPI_2_1_Cora(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(PPI_2_1_Cora, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "tetxtttas"

    @property
    def processed_file_names(self):
        return "PPtI_2_t1ettfdgs"

    def download(self):
        pass

    def process(self):

        name = 'Cora'
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        dataset = Planetoid(path, name)
        data = dataset[0]

        x = data.x.cpu().detach().numpy()
        edge_index = data.edge_index.cpu().detach().numpy()

        # Create graph for easier processing.
        g = Graph(directed=False)
        num_nodes = x.shape[0]

        node_features = {}
        for i in range(num_nodes):
            v = g.add_vertex()
            node_features[v] = x[i]

        rows = list(edge_index[0])
        cols = list(edge_index[1])

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

        data_list = []

        data_new = Data()

        edge_index_1 = torch.tensor(matrix_1).t().contiguous()
        edge_index_2 = torch.tensor(matrix_2).t().contiguous()

        data_new.edge_index_1 = edge_index_1
        data_new.edge_index_2 = edge_index_2

        data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)
        data_new.index_1 = torch.from_numpy(np.array(index_1)).to(torch.int64)
        data_new.index_2 = torch.from_numpy(np.array(index_2)).to(torch.int64)

        data_new.y = data.y

        data_new.train_mask = data.train_mask
        data_new.test_mask = data.test_mask
        data_new.val_mask = data.val_mask

        data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PPI_2_1_Wisconsin(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(PPI_2_1_Wisconsin, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "tetxtttas"

    @property
    def processed_file_names(self):
        return "PPtI_2_t1ettfdgs"

    def download(self):
        pass

    def process(self):

        name = 'wisconsin'
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
        dataset = WebKB(path, name)
        data = dataset[0]

        x = data.x.cpu().detach().numpy()
        edge_index = data.edge_index.cpu().detach().numpy()

        # Create graph for easier processing.
        g = Graph(directed=False)
        num_nodes = x.shape[0]

        node_features = {}
        for i in range(num_nodes):
            v = g.add_vertex()
            node_features[v] = x[i]

        rows = list(edge_index[0])
        cols = list(edge_index[1])

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

        data_list = []

        data_new = Data()

        edge_index_1 = torch.tensor(matrix_1).t().contiguous()
        edge_index_2 = torch.tensor(matrix_2).t().contiguous()

        data_new.edge_index_1 = edge_index_1
        data_new.edge_index_2 = edge_index_2

        data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)
        data_new.index_1 = torch.from_numpy(np.array(index_1)).to(torch.int64)
        data_new.index_2 = torch.from_numpy(np.array(index_2)).to(torch.int64)

        data_new.y = data.y

        data_new.train_mask = data.train_mask
        data_new.test_mask = data.test_mask
        data_new.val_mask = data.val_mask

        data_list.append(data_new)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key in [
            'edge_index_1', 'edge_index_2', 'index_1', 'index_2'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data



class Net(torch.nn.Module):
    def __init__(self, data):
        super(Net, self).__init__()
        self.data=data
        input_dim = self.data.x.shape[1]
        dim = 256
        self.conv_1_1 = GCNConv(input_dim, dim)
        self.conv_1_2 = GCNConv(input_dim, dim)

        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.conv_2_1 = GCNConv(dim, dim)
        self.conv_2_2 = GCNConv(dim, dim)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.mlp = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, 7))
        self.apply(self._init_weights)

    # At initialization of the weights according to a normal distribution
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self):
        x, edge_index_1, edge_index_2 = self.data.x, self.data.edge_index_1, self.data.edge_index_2

        index_1, index_2 = self.data.index_1, self.data.index_2

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
        x = self.mlp(torch.cat([x_1, x_2], dim=1))
        return F.log_softmax(x, dim=1)


class SpeqNet21_for_NC(object) : 

    def __init__(self, dataset_name , num_epochs_training = 200) :

        self.dataset_name = dataset_name
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        if self.dataset_name == 'Cora' :
            self.dataset = PPI_2_1_Cora(path, transform=MyTransform())

        elif self.dataset_name == "wisconsin" : 
            self.dataset = PPI_2_1_Wisconsin(path, transform=MyTransform())

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        self.data = self.dataset[0].to(self.device)

        #Make a copy 
        self.data2 = self.dataset[0].to(self.device)

        #Model
        self.model= Net(self.data).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-3)
        self.num_epochs_training = num_epochs_training

        #Initialization
        self.F1_all=[]
        self.Epochs = []
        self.Training_time=[]
        self.Training_loss=[]
        self.Validation_loss = []



    def train(self,i):
        self.model.train()
        self.optimizer.zero_grad()
        if self.dataset_name == 'Cora' :
            loss = F.nll_loss(self.model()[self.current_data.train_mask], self.current_data.y[self.current_data.train_mask])
        elif self.dataset_name == 'wisconsin' : 
            if self.current_data == self.data2 : 
                loss = F.nll_loss(self.model()[self.current_data.train_mask], self.current_data.y[self.current_data.train_mask])
            else:
                loss = F.nll_loss(self.model()[self.current_data.train_mask[:,i]], self.current_data.y[self.current_data.train_mask[:,i]])
        loss.backward()
        self.optimizer.step()
        return loss


    @torch.no_grad()
    def test(self,i):

        self.model.eval()
        logits, accs, F1score = self.model(), [], 0
        idx = 0
        for _, mask in self.current_data('train_mask', 'val_mask', 'test_mask'):

            if self.dataset_name == 'Cora' : 
                pred = logits[mask].max(1)[1]
                true = self.current_data.y[mask]
                acc = pred.eq(self.current_data.y[mask]).sum().item() / mask.sum().item()
            

                if idx == 1 :
                    validation_loss = F.nll_loss(logits[mask], self.current_data.y[mask])
                if idx==2:
                    y_pred = pred.numpy()
                    y_true = true.numpy()
                    F1score=f1_score(y_true, y_pred, average='macro')

            elif self.dataset_name == 'wisconsin': 

                if idx == 0: #training mask
                    if self.current_data == self.data2 : 
                        pred = logits[mask].max(1)[1]
                        true = self.current_data.y[mask]
                        acc = pred.eq(self.current_data.y[mask]).sum().item() / mask.sum().item()
                    else:
                        pred = logits[mask[:,i]].max(1)[1]
                        true = self.current_data.y[mask[:,i]]
                        acc = pred.eq(self.current_data.y[mask[:,i]]).sum().item() / mask[:,i].sum().item()

                if idx == 1 : #validation mask
                    validation_loss = F.nll_loss(logits[mask[:,i]], self.current_data.y[mask[:,i]])
                if idx==2: #testing mask
                    y_pred = pred.numpy()
                    y_true = true.numpy()
                    F1score=f1_score(y_true, y_pred, average='macro')


            accs.append(acc)
            idx += 1
        
        accs.append(F1score)
        accs.append(validation_loss)

        return accs

    def full_training(self, volume_training = False ):
        acc_all = []

        if volume_training == True : 
            self.current_data = self.data2
        else : 
            self.current_data = self.data

        #Early Stopping
        patience = 30
        trigger_value = 0 

        #Aggregate values for plotting
        global_F1=[]


        starting_time = time.time()

        for i in range(1):
            acc_total = 0
            for i in range(10):
                self.F1_all=[]
                self.Epochs = []
                self.Training_time=[]
                self.Training_loss=[]
                self.Validation_loss = []


                best_val_acc = test_acc = 0
                lowest_val_loss  = float('inf')
                for epoch in range(0,self.num_epochs_training):
                    training_loss = self.train(i)
                    train_acc, val_acc, tmp_test_acc, fsc1 , validation_loss= self.test(i)

                    #Get the values to plot every 5th epoch

                    if epoch % 5 == 0:
                        self.F1_all.append(fsc1)
                        self.Epochs.append(epoch)
                        current_time = time.time() - starting_time
                        self.Training_time.append(current_time)
                        self.Training_loss.append(training_loss.item())
                        self.Validation_loss.append(validation_loss.item())


                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        test_acc = tmp_test_acc


                    if validation_loss < lowest_val_loss : 
                        lowest_val_loss = validation_loss
                        trigger_value = 0

                    if validation_loss >= lowest_val_loss : 
                        trigger_value+=1

                        #Early stopping : if the validation don't improve during 30 consecutive epoch : we stop the traning 
                        if trigger_value >= patience : 
                            print('Early Stopping at Epoch', epoch)
                            print(i, f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                                    f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
                            break


                    #print(i, f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                            #f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
                    

                global_F1.append(np.array(self.F1_all).mean())
                acc_total += test_acc * 100

            acc_all.append(acc_total/10)
            final_time = time.time() - starting_time
        
        if volume_training == False : 

            with open('Report Training - SpeqNet21  - '+ self.dataset_name +' - NC.txt', 'w') as f:
                f.write('Number of epochs of training : ' + str(self.num_epochs_training)+'\n')
                f.write('Time of training  : ' + str(final_time) +'\n')    
                f.write('Final Accuracy : ' + str(np.array(acc_all).mean())+'\n')
                f.write('Final Accuracy : ' + str(np.array(global_F1).mean()))


        print(np.array(acc_all).mean(), np.array(acc_all).std())
        print('\n')
        print("Macro F1-score for all the trained models after Early Stopping on Training Set :  \n" )
        print(global_F1)
        return self.F1_all, self.Training_loss, self.Validation_loss, self.Epochs

    def get_training_plots(self):
        #Plotting


        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(30)
        fig.set_figwidth(30)
        fig.suptitle('Training performances of Speqnet-2-1 - on '+ self.dataset_name+' - With Early Stopping')
        axs[0,0].plot(self.Epochs, self.F1_all)
        axs[0,0].set_title('Performance vs Epochs')
        axs[0,0].set_xlabel('Epoch')
        axs[0,1].plot(self.Epochs, self.Training_loss, label='Training Loss')
        axs[0,1].plot(self.Epochs, self.Validation_loss, label='Validation Loss')
        axs[0,1].legend()
        axs[0,1].set_title('Training and Validation Loss vs Epochs')
        axs[0,1].set_xlabel('Epoch')
        axs[1,0]. plot(self.Training_time, self.F1_all)
        axs[1,0].set_title('Performance vs Training Time')
        axs[1,0].set_xlabel('Training Time (s)')
        axs[1,1].plot(self.Training_time, self.Training_loss, label='Training Loss')
        axs[1,1].plot(self.Training_time, self.Validation_loss, label="Validation Loss")
        axs[1,1].set_title('Training and Validation Loss vs Training Time')
        axs[1,1].set_xlabel('Training Time (s)')
        axs[1,1].legend()
        fig.savefig('Benchmark SpeqNet-2-1_Nodes Classification.png')
        plt.show()


    def volume_training(self) : 

        train_indices = (self.data2.train_mask==True).nonzero().flatten()
        unique_y = torch.unique(self.data2.y)
        fraction_remove=0.2 
        self.Vol_F1, self.Vol_TL, self.Vol_VL, Vol_epochs = {}, {},{}, {}
        self.Vol_data = []
        for i in range(0,5):
            # Implement a new model at each time

            self.model.apply(self.model._init_weights)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-3)

            print('\n\n removal iteration', i)
            updated_train_indices = []
            self.data2.train_mask =torch.Tensor([False]*len(self.data2.train_mask))
            self.data2.train_mask = self.data2.train_mask > 0
            for cur_y in unique_y:
                indices_cur_y =   (self.data2.y==cur_y).nonzero().flatten()
                train_indices_y = np.intersect1d(indices_cur_y, train_indices)
                num_elem_from_y_to_remove = math.ceil(fraction_remove*len(train_indices_y))

                train_indices_y_updated = train_indices_y[0:len(train_indices_y) - num_elem_from_y_to_remove]
                updated_train_indices.extend(train_indices_y_updated)

            #train indices after first step of removal
            train_indices = torch.Tensor(updated_train_indices).to(torch.int64)
            self.data2.train_mask[train_indices]= True # updated train mask
            f1_list , trai_loss_list, val_loss_list, epochs_list = self.full_training(volume_training=True)

            #List of lists
            vol_data = self.data2.train_mask.sum().item()
            self.Vol_data.append(vol_data)

            Vol_epochs[vol_data] = epochs_list
            self.Vol_F1[vol_data] = f1_list
            self.Vol_TL[vol_data] = trai_loss_list
            self.Vol_VL[vol_data] = val_loss_list


        # Plotting the result

        fig, axs = plt.subplots(1, 3)
        fig.set_figheight(15)
        fig.set_figwidth(30)
        fig.suptitle('Training performances of SpeqNet - on '+ self.dataset_name+' - With Early Stopping')

        for vol in self.Vol_F1 : 
            axs[0].plot(Vol_epochs.get(vol), self.Vol_F1.get(vol), label=str(vol)+' graphs for training')
            axs[1].plot(Vol_epochs.get(vol), self.Vol_TL[vol], label=str(vol)+' graphs for training')
            axs[2].plot(Vol_epochs[vol], self.Vol_VL[vol], label=str(vol)+' graphs for training')

        axs[0].set_title('Performance vs Epochs')
        axs[0].set_xlabel('Epoch')
        axs[1].set_title('Training Loss vs Epochs')
        axs[1].set_xlabel('Epochs')
        axs[2].set_title('Validation Loss vs Epochs')
        axs[2].set_xlabel('Epochs')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        fig.savefig('Benchmark - SpeqNet - '+ self.dataset_name + ' - Nodes Classification - Volume training.png')
        print('Plot saved')

