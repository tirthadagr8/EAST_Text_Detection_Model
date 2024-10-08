import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset,get_metadata
from model import EAST
from loss import Loss
import os
import time
from datetime import datetime
from utils import display
import numpy as np
from tqdm import tqdm


def train(metadata,vocab,char_to_int,int_to_char, pths_path, batch_size, lr, epoch_iter, interval):
	trainset = custom_dataset(metadata,0.25)
	train_loader = DataLoader(trainset, batch_size=batch_size, \
                                   		shuffle=True, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()
	model.load_state_dict(torch.load(os.path.join(os.getcwd(),'east.pt'),weights_only=False,map_location=device))
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	for epoch in range(epoch_iter):
		model.train()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in (enumerate(tqdm(train_loader))):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

			epoch_loss += loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step(epoch_loss)
		display(gt_geo.permute(1,0,2,3))
		display(pred_geo.permute(1,0,2,3))
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}, lr is {}'.format(epoch_loss.item(), time.time()-epoch_time,scheduler.get_last_lr()[0]))
	torch.save(model.state_dict(),f'east_{str(datetime.now())[:10]}.pt')


if __name__ == '__main__':
	metadata,vocab,char_to_int,int_to_char=get_metadata()
	pths_path      = './pths'
	batch_size     = 1 
	lr             = 1e-3
	epoch_iter     = 600
	save_interval  = 5
	train(metadata,vocab,char_to_int,int_to_char, pths_path, batch_size, lr, epoch_iter, save_interval)	
	
