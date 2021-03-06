import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
#%matplotlib inline

torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

class Net(torch.nn.Module):
	"""docstring for Net"""
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.hidden = torch.nn.Linear(n_feature, n_hidden)
		self.predict = torch.nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		return x

# net = Net(n_feature = 1, n_hidden = 10, n_output = 1)
net = torch.nn.Sequential(
		torch.nn.Linear(1, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1)
	)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.2)
loss_func = torch.nn.MSELoss()

plt.ion()

for t in range(100):

	prediction = net(x)
	print(prediction)
	loss = loss_func(prediction, y)

	optimizer.zero_grad()
	loss.backward()

	if t % 10 == 0:
		plt.cla()
		plt.scatter(x.data.numpy(), y.data.numpy())
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
		plt.text(0.5, 0, 'Loss = %.4f' % loss.data.numpy(), fontdict = {'size' : 20, 'color': 'red'})
		plt.show()
		plt.pause(0.1)

plt.ioff()