# Benchmarkink-SpeqNets
Evauating the SpeqNet model VS GCN on Node Classification and Graph Similarity Learning

The `run.sh` file permit to launch all the training in a row (Upcoming update)

To run a specific training, the command line is : 

`python3 general_training.py --model <model> --dataset <dataset> --task <task_name>`

With :
`<task>` in ('NC', 'GSL')
`<dataset>` in ('Cora', 'wisconsin', 'LINUX', 'AIDS700nef')
`<model>` in ('GCN', 'Speqnet21')
