from    SpecNet_2_1_nc import *
from gcn_nc import *
from gnn_ged import *
from SpecNet_2_1_ged import *
import argparse

parser = argparse.ArgumentParser()



parser.add_argument("--model", help= "Model to benchmark or baseline", type=str, choices=['Speqnet21', 'GCN'] )
parser.add_argument("--dataset", help="name of the dataset", type=str, choices=['Cora', 'wisconsin', 'LINUX','AIDS700nef'])
parser.add_argument("--task", help= "On which task test the model", type=str, choices=['NC', 'GSL'] )

args = parser.parse_args()

name_model = args.model 
dataset_name = args.dataset
task = args.task

if task == "NC" : 
    if name_model == "Speqnet21":
        model = SpeqNet21_for_NC(dataset_name)

    elif name_model == "GCN" : 
        model = GNN_for_NC(dataset_name)
    else : 
        print('The name of the model must be in [Speqnet21, GCN]')
elif task == "GSL" : 
    if name_model == "Speqnet21":
        model = SpeqNet_for_GSL(dataset_name)

    elif name_model == "GCN" : 
        model = GNN_for_GSL(dataset_name)

    else : 
        print('The name of the model must be in [Speqnet21, GCN]')

else : 
    print('The name of the task must be contained in [NC, GSL] : NC = Nodes Classification , GSL = Graph Similarity Learning')


print('Step 1/3 :  Training on the full dataset')
model.full_training()

print('Step 2/3 : Saving of the training plots')
model.get_training_plots()

print('Step 3/3 : Studying the impact of the volume of training data')
model.volume_training()


