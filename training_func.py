


from bindsnet.learning import PostPre
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network import load
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.models import DiehlAndCook2015
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.datasets import MNIST, DataLoader

from torchvision import transforms
from time import time as t
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import draw_weights, plot_spikes
from connections import connect_one_to_one, connect_all_to_all, connect_random






## Training
def training(net, train_dataset, model_name, weights, spikes_mon, n_epochs=1, time=100, batch_size=32, plot=True, gpu=True):

    print("Begin training.\n")
    start = t()

    # inits
    ims = None
    sel_tracker = []

    # training loop
    for epoch in range(n_epochs):

        # reshuffle dataset
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=gpu,
        )

        # iterating over batches
        pbar_training = tqdm(total=len(train_dataset))
        for step, batch in enumerate(train_dataloader):
            
            # plot weights for every batch
            if plot:
                ims = draw_weights(weights, step, ims)
                plt.pause(1)

            # Get next input sample
            inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, 28, 28)}
            if gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # train the network on batch
            net.run(inputs=inputs, time=time)

            # get spks from monitor
            s = spikes_mon.get("s").squeeze() # fun fact, this has the same result as
                                              # spikes_mon.get("s")[:, 0].contiguous() 
            # calc firing rate 
            rates = torch.sum(s, dim=0)/time # rates per neuron per img per batch, has shape [batch_size, n_neurons]
            mean_rates = torch.mean(rates, dim=0)
            max_rates = torch.max(rates, dim=0).values  

            # calc neuron selectivity 
            sel = 1 - mean_rates/(max_rates + 1e-8) # selectivtiy per neurons over batch, has shape [n_neurons]
            sel_tracker.append(torch.mean(sel).cpu().numpy())

            net.reset_state_variables()  # Reset state variables.
            
            pbar_training.set_description_str(
            f"Mean Selectivity of exc. layer: {torch.mean(sel):.3}"
            )
            pbar_training.update(batch_size)
            
        pbar_training.close()

    net.save(f'models/{model_name}.pt')
    plt.close()
    plt.plot(sel_tracker)
    plt.tile("Selectivity")
    plt.show()
    plt.pause(1000)
    print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
    print("Training complete.\n")
