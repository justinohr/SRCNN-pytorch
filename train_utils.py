from math import log10
import torch

def train(model, train_data, loss_func, optimizer, device):
	model.train()
	X, Y = train_data
	X.requires_grad_(True)
	Y.requires_grad_(True)
	X, Y = X.to(device), Y.to(device)
	model.zero_grad()
	output = model(Y)
	loss = loss_func(X, output)
	print("Loss: %lf" % loss.item())
	loss.backward()
	optimizer.step()

def test(model, test_data, loss_func, epoch, device):
	X, Y = test_data
	PSNR = 0
	for i in range(5):
		x, y = X[i].to(device), Y[i].to(device)
		x = x.unsqueeze(0)
		y = y.unsqueeze(0)
		with torch.no_grad():	
			output = model(y)
			PSNR += 10 * log10(1/loss_func(x,output))
	print("%d epoch's PSNR: %lf" %(epoch + 1, PSNR / 5))