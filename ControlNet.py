from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.datasets import MNIST
from bindsnet.learning import PostPre
from bindsnet.encoding import PoissonEncoder

from torchvision import transforms
import torch

from training_func import training


## Params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = device == torch.device("cuda") # tracks if gpu is used
print("Running on Device = ", device)

# data vars
intensity = 128
n_classes = 10

# Network 
model_name = "Control"

# architecture
n_inpt = 784
inpt_shape = (1, 28, 28)
n_neurons = 100

# time params 
time = 100
dt = 1.0

# Layers
input_layer = Input(n=n_inpt, shape=inpt_shape, traces=True)

target_layer = LIFNodes(n=n_neurons, traces=True)

connection = Connection(
    source=input_layer, target=target_layer, update_rule=PostPre, nu=(1e-4, 1e-2)
)

net = Network()
net.add_layer(input_layer, name="X")
net.add_layer(target_layer, name="Y")
net.add_connection(connection, source="X", target="Y")
net.to(device)

# add monitor
spikes_mon = Monitor(net.layers["Y"], state_vars=["s"], time=time)
net.add_monitor(spikes_mon, name="s")


## Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# run training func
training(net, train_dataset, model_name, weights=net.connections[('X', 'Y')].w, spikes_mon=spikes_mon)

