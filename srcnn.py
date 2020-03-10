import torch
import torch.nn as nn


class SRCNN(nn.Module):
	def __init__(self, config):
		super(SRCNN, self).__init__()
		self.convs = self._make_layers(config)

	def forward(self, x):
		return self.convs(x)

	def _make_layers(self, config):
		layers = []
		#ch_in = 3
		ch_in = 1
		for x in config:
			ch_out, k, s, p = x
			layers += [nn.Conv2d(ch_in, ch_out,
								kernel_size=k, stride=s, padding=p),
					  nn.ReLU(inplace=True)]
			ch_in = ch_out
		del layers[-1] 
		'''removing the ReLU activation function
		of third convolution layer'''  
		return nn.Sequential(*layers)