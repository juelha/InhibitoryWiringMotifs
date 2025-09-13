from bindsnet.network import Network
from bindsnet.network import load
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.datasets import MNIST
from bindsnet.learning import PostPre
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.datasets import MNIST, DataLoader

from torchvision import transforms
from time import time as t
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils import draw_weights, plot_spikes
from src.connections import connect_one_to_one, connect_all_to_all, connect_random, connect_distance
from src.training_func import training


## Params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on Device", device)
gpu = device == torch.device("cuda") # tracks if gpu is used

# data vars
intensity = 128
n_classes = 10

# Network 
model_name = "LocalRandom"

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
reduction = None
wmin = 0.0
wmax = 1.0
theta_plus = 0.05
tc_theta_decay = 1e7
inh_thresh = -40.0
exc_thresh = -52.0

# time params 
time = 100
dt = 1.0

## Layers
input_layer = Input(
    n=n_inpt, shape=inpt_shape, traces=True, tc_trace=20.0
)

exc_layer = DiehlAndCookNodes(
    n=n_neurons,
    traces=True,
    rest=-65.0,
    reset=-60.0,
    thresh=exc_thresh,
    refrac=5,
    tc_decay=100.0,
    tc_trace=20.0,
    theta_plus=theta_plus,
    tc_theta_decay=tc_theta_decay,
)

inh_layer = LIFNodes(
    n=n_neurons,
    traces=False,
    rest=-60.0,
    reset=-45.0,
    thresh=inh_thresh,
    tc_decay=10.0,
    refrac=2,
    tc_trace=20.0,
)

# Connections
# I -> Ex
connections = connect_all_to_all(n_inpt, n_neurons)
weights     = np.random.rand(n_inpt, n_neurons).astype(np.float32)  
w           = exc * torch.from_numpy(connections * weights)
input_exc_conn = Connection(
    source=input_layer,
    target=exc_layer,
    w=w,
    update_rule=PostPre, # stdp
    nu=nu,
    reduction=reduction,
    wmin=wmin,
    wmax=wmax,
    norm=norm,
)

# Ex -> Inh
connections = connect_one_to_one(n_neurons, n_neurons)
weights     = np.random.rand(n_neurons, n_neurons).astype(np.float32)  
w           = exc * torch.from_numpy(connections * weights)
exc_inh_conn = Connection(
    source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=exc
)

# Inh -> Ex
connections1 = connect_all_to_all(n_neurons, n_neurons)
connections2 = connect_distance(n_neurons, n_neurons)
weights     = np.random.rand(n_neurons, n_neurons).astype(np.float32)  
w           = -inh * torch.from_numpy((connections1-connections2) * weights)

inh_exc_conn = Connection(
    source=inh_layer, target=exc_layer, w=w, wmin=-inh, wmax=0
)

# Add to network
net = Network()
net.add_layer(input_layer, name="X")
net.add_layer(exc_layer, name="Exc")
net.add_layer(inh_layer, name="Inh")
net.add_connection(input_exc_conn, source="X", target="Exc")
net.add_connection(exc_inh_conn, source="Exc", target="Inh")
net.add_connection(inh_exc_conn, source="Inh", target="Exc")
net.to(device)
print(net.layers)

# add monitor
spikes_mon = Monitor(net.layers["Exc"], state_vars=["s"], time=time)
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
training(net, train_dataset, model_name, weights=net.connections[('X', 'Exc')].w, spikes_mon=spikes_mon)


