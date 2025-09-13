from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015
from bindsnet.datasets import MNIST, DataLoader

from torchvision import transforms
import torch

from training_func import training


## Params
plot = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on Device", device)
gpu = device == torch.device("cuda") # tracks if gpu is used

# data 
intensity = 128
n_classes = 10

# Network 
model_name = "DiehlCook"

# architecture
n_inpt = 784
inpt_shape = (1, 28, 28)
n_neurons = 100

# training params
n_epochs = 1
batch_size = 32

# model params
exc = 22.5
inh = 120
norm = 78.4
nu = (1e-4, 1e-2)
theta_plus = 0.05
exc_thresh = -200

# sim params 
time = 100
dt = 1.0

# Layers
net = DiehlAndCook2015(
    n_inpt=n_inpt,
    n_neurons=n_neurons,
    inpt_shape=inpt_shape,
    dt=dt,
    exc=exc,
    inh=inh,
    nu = nu, 
    norm=norm,
    theta_plus=theta_plus,
    exc_thresh=exc_thresh
)
net.to(device)
print(f"model {model_name}: {net.layers}")

# add monitor
spikes_mon = Monitor(net.layers["Ae"], state_vars=["s"], time=time)
net.add_monitor(spikes_mon, name="s")

## Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../data/MNIST",
    download=True,
   # train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# run training func
training(net, train_dataset, model_name, weights=net.connections[('X', 'Ae')].w, spikes_mon=spikes_mon)
