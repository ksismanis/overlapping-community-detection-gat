import statistics
import time
import nocd
# import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import normalize

# %matplotlib inline

torch.set_default_tensor_type(torch.cuda.FloatTensor)

# f= 'mag_cs'
# loader = nocd.data.load_dataset('data/mag_cs.npz')
# f = 'fb348'
# loader = nocd.data.load_dataset('data/facebook_ego/fb_348.npz')
f= 'mag_chem'
loader = nocd.data.load_dataset('data/mag_chem.npz')

A, X, Z_gt = loader['A'], loader['X'], loader['Z']
N, K = Z_gt.shape

hidden_sizes = [128]    # hidden sizes of the GNN
weight_decay = 1e-2     # strength of L2 regularization on GNN weights
dropout = 0.5           # whether to use dropout
batch_norm = True       # whether to use batch norm
lr = 1e-3               # learning rate
max_epochs = 500        # number of epochs to train
display_step = 25       # how often to compute validation loss
balance_loss = True     # whether to use balanced loss
stochastic_loss = True  # whether to use stochastic or full-batch training
batch_size = 20000      # batch size (only for stochastic training)



x_norm = normalize(X)  # node features
# x_norm = normalize(A)  # adjacency matrix
# x_norm = sp.hstack([normalize(X), normalize(A)])  # concatenate A and X
x_norm = nocd.utils.to_sparse_tensor(x_norm).cuda()
sampler = nocd.sampler.get_edge_sampler(A, batch_size, batch_size, num_workers=5)
# gnn = nocd.nn.GCN(x_norm.shape[1], hidden_sizes, K, batch_norm=batch_norm, dropout=dropout).cuda()
# gnn = nocd.nn.GAT(x_norm.shape[1],hidden_sizes,K,batch_norm=batch_norm,dropout=dropout).cuda()
gnn = nocd.nn.GAT(x_norm.shape[1], hidden_sizes, K,batch_norm=batch_norm,dropout=dropout).cuda()
adj_norm = gnn.normalize_adj(A)
decoder = nocd.nn.BerpoDecoder(N, A.nnz, balance_loss=balance_loss)
opt = torch.optim.Adam(gnn.parameters(), lr=lr)


def get_nmi(thresh=0.5):
    """Compute Overlapping NMI of the communities predicted by the GNN."""
    gnn.eval()
    Z = F.relu(gnn(x_norm, adj_norm))
    Z_pred = Z.cpu().detach().numpy() > thresh
    nmi = nocd.metrics.overlapping_nmi(Z_pred, Z_gt)
    return nmi



def train():
    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = nocd.train.NoImprovementStopping(validation_fn, patience=10)
    model_saver = nocd.train.ModelSaver(gnn)
    for epoch, batch in enumerate(sampler):
        if epoch > max_epochs:
            break
        if epoch % 25 == 0:
            with torch.no_grad():
                gnn.eval()
                # Compute validation loss
                Z = F.relu(gnn(x_norm, adj_norm))
                val_loss = decoder.loss_full(Z, A)
                print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}, nmi = {get_nmi():.2f}')
                
                # Check if it's time for early stopping / to save the model
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
                if early_stopping.should_stop():
                    print(f'Breaking due to early stopping at epoch {epoch}')
                    break
                
        # Training step
        gnn.train()
        opt.zero_grad()
        Z = F.relu(gnn(x_norm, adj_norm))
        ones_idx, zeros_idx = batch
        if stochastic_loss:
            loss = decoder.loss_batch(Z, ones_idx, zeros_idx)
        else:
            loss = decoder.loss_full(Z, A)
        loss += nocd.utils.l2_reg_loss(gnn, scale=weight_decay)
        loss.backward()
        opt.step()

        
    thresh = 0.5
    Z = F.relu(gnn(x_norm, adj_norm))
    Z_pred = Z.cpu().detach().numpy() > thresh
    model_saver.restore()
    nmi = get_nmi(thresh)
    print(f'Final nmi = {nmi:.3f}')
    return(nmi)

res = []
times = []
for loops in range(20):
    print('loop ', loops)
    st = time.time()
    nmi = train()
    et = time.time()
    elapsed_time = et  - st
    res.append(nmi)
    times.append(elapsed_time)
    for layer in gnn.layers:
            layer.reset_parameters()
    if loops > 2:
        avg = statistics.mean(res)
        sd = statistics.stdev(res)
        avg_time  = statistics.mean(times)
        print('average nmi after', loops,'is ', avg)
        print('standard deviation  nmi  after', loops,'is ', sd)
        print('avrate time   after', loops,'is ', avg_time)
        print(f'average_wmi for {f} is: {avg:.3f}, time  is {avg_time:.3f} seconds, deviation is:  {sd:.4f}')

print(f'final average_wmi for {f} is: {avg:.3f}, time  is {avg_time:.3f} seconds, deviation is:  {sd:.4f}')

resultfile = open('gat_6_heads_with_features.txt', "a")
resultfile.writelines(f' {f} \t {avg:.3f} \t {avg_time:.3f}\t {sd:.3f} \n')
resultfile.close()
