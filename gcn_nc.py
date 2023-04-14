import os.path as osp
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
import os 
import math




class Net(torch.nn.Module):
    def __init__(self, dataset, data):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        self.data = data
        self.apply(self._init_weights)

    # At initialization of the weights according to a normal distribution
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self):
        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)



class GNN_for_NC(object) : 
    def __init__(self, dataset, num_epochs_training = 200):
        self.dataset_name = dataset
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        print(path)
        if self.dataset_name == 'Cora' : 
            self.dataset = Planetoid(path, dataset)
        elif self.dataset_name == 'wisconsin' : 
            self.dataset = WebKB(path, dataset) 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.dataset[0].to(self.device)
        #Make a copy 
        self.data2 = self.dataset[0].to(self.device)

        self.model= Net(self.dataset, self.data).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-3)
        self.num_epochs_training = num_epochs_training
        self.F1_all=[]
        self.Epochs = []
        self.Training_time=[]
        self.Training_loss=[]
        self.Validation_loss = []

    def train(self, i):
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
    def test(self, i):
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
            
            with open('Report Training - GCN - '+ self.dataset_name +'- NC.txt', 'w') as f:
                f.write('Dataset  : ' + self.dataset_name +'\n')
                f.write('Number of epochs of training : ' + str(self.num_epochs_training)+'\n')
                f.write('Time of training  : ' + str(final_time) +'\n')    
                f.write('Final Accuracy : ' + str(np.array(acc_all).mean())+'\n')
                f.write('Final F1 Score : ' + str(np.array(global_F1).mean()))

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
        fig.suptitle('Training performances of GCN - on '+ self.dataset_name+' - With Early Stopping')
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
        fig.savefig('Benchmark GCN_Nodes -' + self.dataset_name+ 'Classification.png')
        plt.show()

    def volume_training(self) : 

        train_indices = (self.data2.train_mask==True).nonzero().flatten()
        unique_y = torch.unique(self.data2.y)
        fraction_remove=0.2 
        self.Vol_F1, self.Vol_TL, self.Vol_VL, Vol_epochs = {}, {},{}, {}
        self.Vol_data = []
        for i in range(0,4):
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
        fig.suptitle('Training performances of GCN - on '+ self.dataset_name+' - With Early Stopping')



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
        fig.savefig('Benchmark - GCN - '+ self.dataset_name + ' - Nodes Classification - Volume training.png')
        print('Plot saved')

