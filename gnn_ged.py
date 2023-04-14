from ast import Num
from asyncio.base_tasks import _task_print_stack
from curses.ascii import EM
from multiprocessing.dummy.connection import families
import os.path as osp
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import OneHotDegree
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
import os 
from data_utils_ged import load_ged_dataset
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import NormalizeFeatures
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
import math


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
    def __init__(self, INPUT_DIM, FINAL_DIM,EMBED_DIM ):
        super().__init__()
        self.INPUT_DIM = INPUT_DIM
        self.FINAL_DIM = FINAL_DIM
        self.EMBED_DIM = EMBED_DIM
        self.conv1 = GCNConv(FINAL_DIM+1, 16, cached=True)
        self.conv2 = GCNConv(16, EMBED_DIM , cached=True)
        self.mlp = MLP(EMBED_DIM, 16, 1)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def compute_degree_vector(self, data):
    # Create a tensor of ones with the same size as the edge index
        num_nodes =  self.FINAL_DIM
        deg_onehot = OneHotDegree(num_nodes, cat = False)(data)
        return deg_onehot


    def forward(self, data1, data2):

        #Initialize the feature vector with the degree of each node
        x1, edge_index1 = self.compute_degree_vector(data1).x , data1.edge_index
        x2, edge_index2 = self.compute_degree_vector(data2).x , data2.edge_index
        data1.x = x1
        data2.x = x2
        #x1 = data1.x
        #x2=data2.x
        s0,_= x1.shape
        s1,_=x2.shape
        data1.x = F.pad(data1.x, (0,0,0,self.INPUT_DIM - s0))
        data2.x = F.pad(data2.x, (0,0,0,self.INPUT_DIM - s1 ))
        data1.x = F.relu(self.conv1(data1.x, edge_index1))
        data2.x = F.relu(self.conv1(data2.x, edge_index2))
        data1.x = self.conv2(data1.x, edge_index1)
        data2.x = self.conv2(data2.x, edge_index2)
        embed_batch1=torch.zeros((len(data1), self.EMBED_DIM))
        embed_batch2=torch.zeros((len(data1), self.EMBED_DIM))
        cur_idx1 = 0
        cur_idx2 = 0
        for i in range(len(data1)) : 
            n1, n2 = data1[i].num_nodes, data2[i].num_nodes
            mat1 = data1.x[cur_idx1 : cur_idx1+n1, :]
            mat2 = data2.x[cur_idx2 : cur_idx2+n2, :]
            embed_graph1 = mat1.sum(dim=0)
            embed_graph2 = mat2.sum(dim=0)
            embed_batch1[i] = embed_graph1
            embed_batch2[i] = embed_graph2
            cur_idx1 = cur_idx1+n1
            cur_idx2 = cur_idx2 + n2
        #Agregate the representation of the two graph by taking the sum of the embeddings 
        X = embed_batch1 + embed_batch2
        dist = self.mlp(X)
        return dist

class GNN_for_GSL(object) : 
    def __init__(self, dataset) : 
        self.name= dataset

        self.idx_training = self.get_subsets(self.name,1)
        if self.name == 'LINUX' : 
            self.idx_val = [x for x in range(601, 800)]
        elif self.name == 'AIDS700nef' : 
            self.idx_val = [x for x in range(421, 560)]

        self.path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', self.name)
        self.dataset = GEDDataset(self.path, self.name, train=True, transform=NormalizeFeatures())
        self.dataset2 = GEDDataset(self.path, self.name, train=True, transform=NormalizeFeatures())
        self.test_dataset = GEDDataset(self.path, self.name, train=False,transform=NormalizeFeatures() )
        self.val_dataset = self.dataset[self.idx_val]

        
        self.FINAL_DIM = max([data.num_nodes for data in self.dataset])
        self.INPUT_DIM = (self.FINAL_DIM )*10
        self.EMBED_DIM = 20

        self.model = Net(self.INPUT_DIM, self.FINAL_DIM,self.EMBED_DIM)
        self.loss_funct = MSELoss(reduce=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-3)
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

    def get_target_vector(self, dataset, batch1, batch2) : 
        T=torch.zeros((len(batch1), 1))
        for e in range(len(batch1)):
            dist = dataset.ged[batch1[e].i, batch2[e].i]
            T[e][0] = dist
        return T
    

    def train(self, batch1, batch2) :  
        self.model.train() 
        self.optimizer.zero_grad()
        predict = self.model(batch1, batch2)
        target = self.get_target_vector(self.dataset, batch1, batch2) #.detach().cpu().numpy()
        loss = self.loss_funct(predict, target)
        loss = loss.squeeze(dim=1).sum()
        loss.backward()
        target = target.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()
        train_rmse = math.sqrt(np.square(np.subtract(target, predict)).mean())
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

        #Validation set
        val_rmse_epoch = 0
        val_loss_epoch = 0
        impl = 0
        for val_batch in self.val_loader : 
            for ref_batch in self.train_loader1 : 
                if len(val_batch) == batch_size and len(ref_batch) == batch_size : 
                    impl +=1
                    pred = self.model(val_batch, ref_batch)
                    true = self.get_target_vector(self.dataset, val_batch, ref_batch)
                    
                    loss_val = self.loss_funct(pred, true)
                    loss_val = loss_val.squeeze(dim=1).sum()
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
        for test_batch in self.test_loader : 
            for ref_batch in self.train_loader1 :
                if len(test_batch)== batch_size and len(ref_batch)== batch_size :  
                    impl +=1
                    pred = self.model(test_batch, ref_batch).detach().cpu().numpy()
                    true = self.get_target_vector(self.dataset, test_batch, ref_batch).detach().cpu().numpy()
                    pred = np.squeeze(pred, axis=1)
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
            self.current_dataset = self.dataset2
        else : 
            self.current_dataset = self.dataset
        print('Volume of training data ', len(self.idx_training))
        self.train_dataset = self.current_dataset[self.idx_training]

        self.train_loader1 = DataLoader(self.train_dataset, batch_size= batch_size, shuffle = True)
        self.train_loader2= DataLoader(self.train_dataset, batch_size= batch_size, shuffle= True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True )
        
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
        for i in range(200) :
            for batch1 in self.train_loader1 : 
                for batch2 in self.train_loader2 :
                    if len(batch1) == batch_size and len(batch2)==batch_size : 
                        self.epoch += 1 
                        train_loss, train_rmse = self.train(batch1, batch2)

                        if self.epoch % 5 == 0:
                            
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
                                        
                


                        print('Epoch : ', self.epoch, ' out of ', len(self.train_loader2)**2  )
                        print('Train Loss : ', train_loss.item())

        
        return self.RMSE_Test, self.Train_Loss, self.Validation_Loss, self.Epochs, self.F1_K_Test
                        


    def get_training_plots(self) : 

        #Plotting
        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(30)
        fig.set_figwidth(30)
        fig.suptitle('Training performances of GCN - on '+ self.name+' - With Early Stopping')
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
        fig.savefig('Benchmark GCN -' + self.name+ '- Graph Similarity Learning.png')
        plt.show()

    def volume_training(self) : 
        self.Vol_RMSE, self.Vol_F1, self.Vol_TL, self.Vol_VL, self.Vol_epochs = {}, {},{},{}, {}
        self.Vol_data = []
        for ratio in range(1,5) :
            ratio = ratio/10
            #Initalize the weights of the model
            self.model.apply(self.model._init_weights)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-3)



            self.idx_training = self.get_subsets(self.name,ratio)
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
        fig.suptitle('Training performances of GCN - on '+ self.name+' - With Early Stopping for Graph Similarity Learning')



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
