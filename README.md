# InhibitoryWiringMotifs

![](https://github.com/juelha/InhibitoryWiringMotifs/blob/main/figs/dc.png)

### About
- Project investigating how inhibitory wiring motifs change the learned representations of the excitatory layer 
- Winning Contribution for the Spiking Neural Networks Hackathon at the University of Osnabrück 2025
- For simulating SNNs I used [BindsNET](https://github.com/BindsNET/bindsnet)
- Check out the [presentation](https://github.com/juelha/InhibitoryWiringMotifs/blob/main/presentation.pdf) 


### Experiment Setup

I chose to have a two-layer set up which also complies with Dale’s law since there is a separate excitatory (red) and inhibitory (blue) population.
The independent variable is the wiring motif between the excitatory and inhibitory layer, which is fixed for each experiment. 


<img align="right" src="figs\experiment_setup.png" width="250">



- E-Layer: [DiehlAndCookNodes()](https://github.com/BindsNET/bindsnet/blob/master/bindsnet/network/nodes.py#L981)

- I-Layer: [LIFNodes()](https://github.com/BindsNET/bindsnet/blob/master/bindsnet/network/nodes.py#L418)

- fixed E→I, E←I connections

- train E-Layer with STDP


As data I use MNIST encoded with a [PoissonEncoder()](https://github.com/BindsNET/bindsnet/blob/master/bindsnet/encoding/encoders.py#L88).


### Results 

#### Control Setup

<img align="left" src="figs\control_setup.png" width="100">

<img align="right" src="figs\stdp_only.png" width="150">

#### Wiring Motif: One to One

<img align="left" src="figs\one_to_one_setup.png" width="100">

<img align="left" src="figs\one-to-one-e.png" width="100">

<img align="left" src="figs\one-to-one-i.png" width="100">

<img align="left" src="figs\one-to-one.png" width="100">


#### Wiring Motif: random (symmetrical)

<img align="left" src="figs\sym_rand_setup.png" width="100">

<img align="left" src="figs\sym_rand-e.png" width="100">

<img align="left" src="figs\sym_rand-i.png" width="100">

<img align="left" src="figs\sym_rand.png" width="100">

#### Wiring Motif: random 

<img align="left" src="figs\rand_setup.png" width="100">

<img align="left" src="figs\rand-e.png" width="100">

<img align="left" src="figs\rand-i.png" width="100">

<img align="left" src="figs\rand.png" width="100">

#### Wiring Motif: Diehl & Cook, 2015 

<img align="left" src="figs\dc_setup.png" width="100">

<img align="left" src="figs\dc-e.png" width="100">

<img align="left" src="figs\dc-i.png" width="100">

<img align="left" src="figs\dc.png" width="100">


#### Wiring Motif: local & random

<img align="left" src="figs\local-rand-setup.png" width="100">

<img align="left" src="figs\local-e.png" width="100">

<img align="left" src="figs\local-i.png" width="100">

<img align="left" src="figs\local.png" width="100">

#### Wiring Motif: local & random

<img align="left" src="figs\dist_setup.png" width="100">

<img align="left" src="figs\dist-e.png" width="100">

<img align="left" src="figs\dist-i.png" width="100">

<img align="left" src="figs\dist.png" width="100">



### Special Requirements 
Install BindsNET with 

``!pip install numpy scipy matplotlib git+https://github.com/BindsNET/bindsnet.git``

(should automatically install torch) 
