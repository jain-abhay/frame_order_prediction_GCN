from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from model import *
import uuid
from scipy import sparse
import torchvision.utils as vutils
from tqdm import tqdm
import os
import scipy.sparse as sp
import cv2
from glob import glob
from flow_utils import plot
from torchvision.utils import flow_to_image 
UNKNOWN_FLOW_THRESH = 1e7

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--m_pos', type=int, default=3, help='postional_encoding')
parser.add_argument('--test_epoch', type=int, default=50, help='postional_encoding')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--save_dir', default='/results/images', help='saving results')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* feat_extract.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--phase', type=str, default='train', help='which phase: test or train')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def flow_error(tgt, pred):
    smallflow = 1e-6
    
    stu = tgt[:,:,0]
    stv = tgt[:,:,1]
    su = pred[:,:,0]
    sv = pred[:,:,1]



    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = (torch.absolute(stu) > smallflow) | (torch.absolute(stv) > smallflow)
    index_su = su[ind2]
    index_sv = sv[ind2]
    an = 1.0 / torch.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / torch.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    epe = torch.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[ind2]
    mepe = torch.mean(epe)
    return mepe

def visualize(imfeat, tgt_flow, pred_flow, phase, epoch, size):
	if not os.path.isdir('results'):
		os.makedirs('results')

	rows,cols = size


	# inp_im1 =  imfeat[:,:,2:5].cpu().detach().reshape((rows,cols,3))*255.0
	# inp_im2 =  imfeat[:,:,2:5].cpu().detach().reshape((rows,cols,3))*255.0
	inp_im1 =  imfeat[0,:,2:5].cpu().detach().reshape((rows,cols,3))
	inp_im2 =  imfeat[1,:,2:5].cpu().detach().reshape((rows,cols,3))

	#### include the code to visualize the estimated flow ######
	tgt_flow  = tgt_flow.cpu().detach().reshape((rows,cols,2))
	pred_flow = pred_flow.cpu().detach().reshape((rows,cols,2))
	tgt_flow  = torch.permute(tgt_flow, [2, 1, 0])
	pred_flow = torch.permute(pred_flow, [2, 1, 0])

	flow_imgs_tgt  = flow_to_image(tgt_flow)
	flow_imgs_pred = flow_to_image(pred_flow)

	plot([[torch.permute(inp_im1, [2, 0, 1]), torch.permute(inp_im2, [2, 0, 1])], [flow_imgs_tgt, flow_imgs_pred]], save_path='results/{}.png'.format(epoch))
	# n_cols  = 4
	# n_rows  = 1
	# height, width, channel  = inp_im1.shape
	# collage = np.zeros((n_rows * height, n_cols* width, channel))
	# collage[:height, :width, :] = inp_im1
	# collage[:height, width:2*width, :] = inp_im2
	# collage[:height, 2*width:3*width, :] = pred_flow
	# collage[:height, 3*width:4*width, :] = tgt_flow

	# if not os.path.isdir(os.path.join('experiments',phase,'results_full')):
	# 	os.makedirs(os.path.join('experiments',phase,'results_full'))
	# save_path = os.path.join('experiments',phase,'results_full','epoch_'+str(epoch)+'.png')
	# cv2.imwrite(save_path, collage)
	


cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)


checkpt_file = 'feat_extract.pt'


rows, cols = 200,200
adj = torch.randn(1,40000,40000).to(device)
nadj = adj.shape[-1]

# Write load model code if already exists


feat_extract =  GCNOF(nfeat=7,
                nlayers=args.layer,
                nhidden=args.hidden,
                nfinal=2,
                nadj = nadj,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                variant=args.variant).to(device)

feat_extract.load_state_dict(torch.load(checkpt_file))

print('-------GCNOF loaded successfully---------')

frame_order =  FOP(nfeat=7,
                nlayers=args.layer,
                nhidden=args.hidden,
                nfinal=12,
                nadj = nadj,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                variant=args.variant).to(device)

print('-------FOP loaded successfully---------')





optimizer = optim.Adam([
                        {'params':feat_extract.params1,'weight_decay':args.wd1},
                        {'params':feat_extract.params2,'weight_decay':args.wd1},
                        {'params':feat_extract.params3,'weight_decay':args.wd2},
                        {'params':frame_order.params2,'weight_decay':args.wd1},
                        {'params':frame_order.params3,'weight_decay':args.wd2}
                        ],lr=args.lr)

print('-------Optimizers initialized successfully---------')

loss_crit = nn.CrossEntropyLoss()


if args.phase == 'train':
	
	# dataloader = CustomData(path_img='/home/cvig/Documents/Clip_Ordering_2/data/MPI-Sintel-complete/training/clean/*',\
	# 				path_flow='/home/cvig/Documents/Clip_Ordering_2/data/MPI-Sintel-complete/training/flow/*/*',
	# 				)
	# train_dataloader = DataLoader(dataloader, batch_size=1, shuffle=True, num_workers=16, prefetch_factor=4)

	# if os.path.isdir(checkpt_fold):
	# 	feat_extract.load_state_dict(torch.load(checkpt_fold +'feat_extract_'+str(len(glob(checkpt_fold+'/*')))+'.pt'))
	# else:
 #    		os.makedirs(checkpt_fold)

	t_total = time.time()
	feat_extract.eval()
	frame_order.train()

	for epoch in range(args.epochs):
		print('-------training started---------')
		# tq = tqdm(train_dataloader)

		imfeat = torch.randn(4,40000,7).to(device)
		target = torch.FloatTensor([[0,0,0,1,0,0,0,0,0,0,0,0]]).to(device)
		target = target.softmax(dim=1)

			
	
		optimizer.zero_grad()

		

		with torch.no_grad():
			pred= feat_extract(imfeat,adj)

		pred_seq = frame_order(pred, adj)

		

		loss_train = loss_crit(pred_seq, target)


		loss_train.backward()
		optimizer.step()
			
		
			
		print('Train loss: {}'.format(loss_train.item()))





