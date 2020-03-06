from srcnn import SRCNN
from train_utils import train, test
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
import random, os

training_file = os.listdir('./SR_dataset/91')
testing_file = os.listdir('./SR_dataset/Set5')

BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 1000000
PRINT_STEP = 10
UPSCALING_FACTOR = 2 # factor of 2,3,4 are allowed

train_transform = transforms.Compose([
				transforms.RandomResizedCrop(32),
				transforms.RandomHorizontalFlip()])

def generate_data(mode):
	if mode == 'train':
		X = []
		Y = []
		for i in range(BATCH_SIZE):
			img = Image.open('./SR_dataset/91/' + random.choice(training_file))
			img = train_transform(img)

			flawed = img.resize((16,16))
			flawed = flawed.resize((32,32), Image.BICUBIC)

			#X.append(transforms.ToTensor()(transforms.CenterCrop(20)(img)))
			X.append(transforms.ToTensor()(img))
			Y.append(transforms.ToTensor()(flawed))
		return (torch.stack(X), torch.stack(Y))
	else:
		X = []
		Y = []
		for file in testing_file:
			img = Image.open('./SR_dataset/Set5/' + file)
			X.append(transforms.ToTensor()(img))

			flawed = img.resize((int(img.size[0]/2), int(img.size[1]/2)))
			flawed = flawed.resize(img.size, Image.BICUBIC)
			Y.append(transforms.ToTensor()(img))
		return (X,Y)

def main():
	config = [(64, 9, 1, 4), (32, 1, 1, 0), (3, 5, 1, 2)]
	#config = [(64, 9, 1, 0), (32, 1, 1, 0), (3, 5, 1, 0)]
	model = SRCNN(config).to(DEVICE)
	loss_function = nn.MSELoss(reduction='sum')
	optimizer = optim.Adam(model.parameters())
	test_data = generate_data('test')
	for epoch in range(EPOCH):
		train_data = generate_data('train')
		train(model, train_data, loss_function, optimizer, DEVICE)
		test(model, test_data, loss_function, epoch, DEVICE)

if __name__ == "__main__":
	main()