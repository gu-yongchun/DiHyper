import configargparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric

from model import DiHyper

# from models import   GCNModel
from scipy.sparse import coo_matrix
#from data import read_data, read_data_pub

from src import process_magnetic_laplacian_sparse
import datasets_direction
import datasets

import utils

from scipy import sparse
from scipy.sparse.linalg import lobpcg
from ufg import get_operator
from ufg import scipy_to_torch_sparse

import os

import time

#%% 函数
def vertex_degree(edge_index, num_nodes):
    # incident matrix construction
    row, col = edge_index
    size_col = max(col) + 1
    H = coo_matrix((np.ones(len(edge_index.T)), (row, col)), shape=(num_nodes, size_col), dtype=np.float32).tocsr()
    degree = torch.from_numpy(np.sum(np.abs(H), axis = 1)).float()
    return degree

@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, loss_fn=None, return_out=False):
    model.eval()
    out = model(data)
    out = F.log_softmax(out, dim=1)

    train_acc = evaluator.eval(data.y[split_idx['train']], out[split_idx['train']])['acc']
    valid_acc = evaluator.eval(data.y[split_idx['valid']], out[split_idx['valid']])['acc']
    test_acc = evaluator.eval(data.y[split_idx['test']], out[split_idx['test']])['acc']

    ret_list = [train_acc, valid_acc, test_acc]

    # Also keep track of losses
    if loss_fn is not None:
        train_loss = loss_fn(out[split_idx['train']], data.y[split_idx['train']])
        valid_loss = loss_fn(out[split_idx['valid']], data.y[split_idx['valid']])
        test_loss = loss_fn(out[split_idx['test']], data.y[split_idx['test']])
        ret_list += [train_loss, valid_loss, test_loss]

    if return_out:
        ret_list.append(out)

    return ret_list

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#%% 参数设置
parser = configargparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--config', is_config_file=True)

# Dataset specific arguments
parser.add_argument('--method', default='DiHyper', help='model type')
parser.add_argument('--dname', default='cora')
parser.add_argument('--second_name', type=str, default='cora')
parser.add_argument('--res_root', type=str, default='temp') #结果保存文件夹


parser.add_argument('--feature_noise', default='1', type=str, help='std for synthetic feature noise')
parser.add_argument('--data_dir', type=str, default='data_data_dir')
parser.add_argument('--raw_data_dir', type=str, default='data/original_data/cocitation/')

parser.add_argument('--directed', type=bool, default=True)
parser.add_argument('--train_prop', type=float, default=0.5)
parser.add_argument('--valid_prop', type=float, default=0.25)
parser.add_argument('--exclude_self', action='store_true', help='whether the he contain self node or not')
parser.add_argument('--normtype', default='all_one', choices=['all_one','deg_half_sym'])
parser.add_argument('--add_self_loop', action='store_false')

# Training specific hyperparameters
parser.add_argument('--epochs', default=500, type=int)
# Number of runs for each split (test fix, only shuffle train/val)
parser.add_argument('--runs', default=10, type=int)
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--input_dropout', default=0.2, type=float)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--wd', default=0.005, type=float)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--ablation', type=str, default='full')

# For saving porpuse
parser.add_argument('--config_number', type=int, default=0)
parser.add_argument("--save_result", action="store_true", default=False)

# Model common hyperparameters
# parser.add_argument('--All_num_layers', default=2, type=int, help='number of basic blocks')
# parser.add_argument('--MLP_num_layers', default=2, type=int, help='layer number of mlps')
# parser.add_argument('--activation', default='relu', choices=['Id','relu', 'prelu'])
# parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
parser.add_argument('--hidden', default=64, type=int, help='hidden dimension of model')
parser.add_argument('--Classifier_num_layers', default=2, type=int)  # How many layers of decoder
parser.add_argument('--Classifier_hidden', default=64, type=int)  # Decoder hidden units
parser.add_argument('--normalization', default='ln', choices=['bn','ln','None'])


# Args for GeDi
parser.add_argument('--other_complex', action='store_true', default=False)
parser.add_argument('--nconv', default=4, type=int)
parser.add_argument('--connection', type=bool, default=True)

# Args for Framelet
parser.add_argument('--FrameType', type=str, default='Haar',
                    help='frame type (default: Haar)')
parser.add_argument('--Lev', type=int, default=2,
                    help='level of transform (default: 2)')
parser.add_argument('--s', type=float, default=2,
                    help='dilation scale > 1 (default: 2)')
parser.add_argument('--n', type=int, default=2,
                    help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')

parser.add_argument('--alpha', type=float, default=0.9, help='alpha_l')
parser.add_argument('--gamma', type=float, default=0.9, help='beta_l')
parser.add_argument('--lamda', type=float, default=0.6, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')

# f'{res_root}/all_args_{args.dname}.csv'
args = parser.parse_args()
print(args)
#     Use the line below for notebook
args.data_dir = f'{args.data_dir}/{args.dname}'  
"固定随机种子"
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.set_default_dtype(torch.float32)

#%% 读取数据
device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
print(device)
if args.method not in ['HyperGCN', 'HyperSAGE']:
    transform = torch_geometric.transforms.Compose([datasets.AddHypergraphSelfLoops()])
else:
    transform = None

"读取数据"
root = args.data_dir
name = args.dname
path_to_download = args.raw_data_dir
feature_noise = args.feature_noise
transform = transform
second_name = args.second_name  


'有向否'
if args.directed:
    print('directed')
    Data = datasets_direction.HypergraphDataset(root=args.data_dir, name=args.dname, path_to_download=args.raw_data_dir,
    feature_noise=args.feature_noise, transform=transform, second_name=args.second_name)
elif not args.directed:
    print('undirected')
    Data = datasets.HypergraphDataset(root=args.data_dir, name=args.dname, path_to_download=args.raw_data_dir,
    feature_noise=args.feature_noise, transform=transform, second_name=args.second_name)

data = Data.data
if data.x is None:
    data.x = vertex_degree(data.edge_index, data.num_nodes)

'有向图拉普拉斯矩阵'
edge_index, norm_real, norm_imag = process_magnetic_laplacian_sparse(edge_index=data.edge_index, x_real=data.x, edge_weight=data.edge_weight, \
    normalization = 'sym', num_nodes=data.num_nodes,return_lambda_max = False)
args.num_nodes = data.num_nodes         
num_nodes = args.num_nodes

#%% famlet_data矩阵处理    
L = sparse.coo_matrix((1j*norm_imag.cpu() + norm_real.cpu(),
                        (edge_index[0,:].cpu(),  edge_index[1,:].cpu())
                        ), 
                      shape=(num_nodes, num_nodes))
# print("L.shape:", L.shape)
lobpcg_init = np.random.rand(num_nodes, 1)  # [2708,1]
# print("lobpcg_init.shape:", lobpcg_init.shape)
lambda_max, _ = lobpcg(L, lobpcg_init)
lambda_max = lambda_max[0]

# extract decomposition/reconstruction Masks
FrameType = args.FrameType

if FrameType == 'Haar':
    D1 = lambda x: np.cos(x / 2)
    D2 = lambda x: np.sin(x / 2)
    DFilters = [D1, D2]
    RFilters = [D1, D2]
elif FrameType == 'Linear':
    D1 = lambda x: np.square(np.cos(x / 2))
    D2 = lambda x: np.sin(x) / np.sqrt(2)
    D3 = lambda x: np.square(np.sin(x / 2))
    DFilters = [D1, D2, D3]
    RFilters = [D1, D2, D3]
elif FrameType == 'Quadratic':  # not accurate so far
    D1 = lambda x: np.cos(x / 2) ** 3
    D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
    D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
    D4 = lambda x: np.sin(x / 2) ** 3
    DFilters = [D1, D2, D3, D4]
    RFilters = [D1, D2, D3, D4]
else:
    raise Exception('Invalid FrameType')

Lev = args.Lev  # level of transform
s = args.s  # dilation scale
n = args.n  # n - 1 = Degree of Chebyshev Polynomial Approximation

J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
r = len(DFilters)

# get matrix operators
d = get_operator(L, DFilters, n, s, J, Lev)  # J传入的是nan?
# enhance sparseness of the matrix operators (optional)
# d[np.abs(d) < 0.001] = 0.0
# store the matrix operators (torch sparse format) into a list: row-by-row
d_list_real = list()
d_list_imag = list()
if 'full' in args.ablation: #消融实验
    for i in range(r):
        for l in range(Lev):
            d_list_real.append(scipy_to_torch_sparse(d[i, l].real).to(device))
            d_list_imag.append(scipy_to_torch_sparse(d[i, l].imag).to(device))
elif 'wolow' in args.ablation:
    for l in range(Lev):
        d_list_real.append(scipy_to_torch_sparse(d[1, l].real).to(device))
        d_list_imag.append(scipy_to_torch_sparse(d[1, l].imag).to(device))
elif 'wohig' in args.ablation:
    for l in range(Lev):
        d_list_real.append(scipy_to_torch_sparse(d[0, l].real).to(device))
        d_list_imag.append(scipy_to_torch_sparse(d[0, l].imag).to(device))        


# data.adj = adj
data.d_list_real = d_list_real
data.d_list_imag = d_list_imag
args.r = r
data = data.to(device)

#%% 数据划分
"数据划分"
# Get splits
split_idx_lst = []
spilt_dir = f'{path_to_download}{args.dname}/split'
if not os.path.isdir(spilt_dir): #判断文件夹是否存在
    os.makedirs(spilt_dir) #创建文件夹
    # files = os.listdir(spilt_dir) #文件夹下所有文件名

for run in range(args.runs): #切分数据集
    entry = f'split{run}' #切分文件名
    spilt_file = os.path.join(spilt_dir, entry) # 构建完整路径
    # print(spilt_file)
    if os.path.exists(spilt_file): #判断文件是否存在
        split_idx = torch.load(spilt_file)
        print(spilt_file)
    else:
        if args.dname in ['WikiCS']:
            split_idx = utils.rand_train_test_idx_2(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop, balance=True) 
        else:
            split_idx = utils.rand_train_test_idx_2(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop, balance=False) 
    split_idx_lst.append(split_idx)

#%% 模型           
"加载模型"
# define the model
if args.method == 'DiHyper':
    model = DiHyper(K=1, num_features=data.num_features, hidden=args.hidden, label_dim=data.num_classes,     # hidden=256 dropout= 0.3
                        i_complex = False,  layer=args.nconv, other_complex=args.other_complex, edge_index=edge_index,\
                       norm_real=norm_real, norm_imag=norm_imag, dropout=args.dropout, gcn=False, args=args)
else:
    raise ValueError(f'Undefined model name: {args.method}')
# ####如果模型已经训练好了
# for run in range(args.runs):
#     model_name = f'model{run}' # 模型文件名
#     '如果模型已经存在'
#     if os.path.exists(model_name): #判断文件是否存在
#         model = torch.load(model_name)

"模型参数统计"
if not args.method == 'PhenomNN':
    model = model.to(device)
    num_params = count_parameters(model)
print("# Params:", num_params)

"加载训练日志"
logger = utils.Logger(args.runs, args) 

loss_fn = nn.NLLLoss() #损失函数
evaluator = utils.NodeClsEvaluator() #度量函数

#%% 训练模型
'模型文件夹是否存在'
model_dir = f'{path_to_download}{args.dname}/model'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir) #创建文件夹
'数据划分文件夹是否存在'
split_dir = f'{path_to_download}{args.dname}/split'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir) #创建文件夹

runtime_list = []
for run in range(args.runs):
    
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    best_val = float('-inf')

    #optimizer = Adafactor(model.parameters(), weight_decay=0)#, lr=0.01, relative_step= False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print('_______________________________________________________')
    "开始训练"
    start_time = time.time()
    for epoch in range(args.epochs):
        # Training loop
        model.train()
        optimizer.zero_grad()
        out = model(data)    #(data)
        out = F.log_softmax(out, dim=1)
        loss = loss_fn(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        
        # Evaluation and logging
        result = evaluate(model, data, split_idx, evaluator, loss_fn)
        logger.add_result(run, *result[:3]) #添加结果到日志
        if epoch % args.display_step == 0 and 100 * result[1] > best_val:
            print(f'Run: {run:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Train Loss: {loss:.4f}, '
                  f'Valid Loss: {result[4]:.4f}, '
                  f'Test Loss: {result[5]:.4f}, '
                  f'Train Acc: {100 * result[0]:.2f}%, '
                  f'Valid Acc: {100 * result[1]:.2f}%, '
                  f'Test Acc: {100 * result[2]:.2f}%')
            best_val = 100 * result[1]
            model_best = model
            model_acc = f'{100 * result[2]:.2f}'


    end_time = time.time()
    runtime_list.append(end_time - start_time)
    
    if args.save_result:
        path_split = split_dir + f'/split{model_acc}_{run}.torch'
        torch.save(split_idx, path_split) #保存划分结果
        torch.save(model_best, model_dir + f'/model{model_acc}_{run}.torch') #保存划分结果
        
# logger.print_statistics(args=args)

## Save results ###
avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

#
best_val, best_test = logger.print_statistics()
# with open(model_dir + f'{best_test.mean():.3f}.csv', 'w') as write_obj:
#     for run in range(args.runs):
#         line = f'split_{run}\t{best_test[run]}\n'
#         write_obj.write(line)
#     write_obj.write(str(args))    

res_root = args.res_root
if not os.path.isdir(res_root):
    os.makedirs(res_root)
#
filename = f'{res_root}/{args.second_name}.csv'
print(f"Saving results to {filename}")
with open(filename, 'a+') as write_obj:
    cur_line = f'{args.method}\t'
    cur_line += f'{best_val.mean():.3f}±{best_val.std():.3f}\t'
    cur_line += f'{best_test.mean():.3f}±{best_test.std():.3f}\t'
    cur_line += f'{num_params}, {avg_time:.2f}s\t' 
    cur_line += f'{avg_time//60}min{(avg_time % 60):.2f}s\t'
    cur_line += f'{str(args)}\t' + '\n'
    write_obj.write(cur_line)
# #
#     all_args_file = f'{res_root}/all_args_{args.dname}.csv'
#     with open(all_args_file, 'a+') as f:
#         f.write(str(args))
#         f.write('\n')
# #    
print('All done! Exit python code')




















