# InhibitoryWiringMotifs

> Project investigating how inhibitory wiring motifs change the learned representations of the excitatory layer 


### About
- Winning Contribution for the Spiking Neural Networks Hackathon at the University of Osnabrück 2025
- For simulating SNNs I used [BindsNET](https://github.com/BindsNET/bindsnet)
- Check out the [presentation](https://github.com/juelha/InhibitoryWiringMotifs/blob/main/presentation.pdf)
---
### Research Question
How can inhibitory wiring motifs change the learned representations of the excitatory layer?

➢ The connections between the inhibitory and excitatory layer can help reducing redundancy in weights and increase sparsity in the excitatory layer’s activity. When the strength of the inhibitory connections increases with distance, learned representations can be pushed into clusters.

| Control |[Diehl & Cook, 2015](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full) wiring motif | [Hazan et al., 2018](https://arxiv.org/abs/1807.09374) wiring motif |
| ------------- | ------------- | ------------- |
| <img src="figs\stdp_only.png" width="300" /> |  <img src="figs\dc.png" width="300" /> | <img src="figs\dist.png" width="300" /> | 

---
### Experiment Setup

<img align="right" src="figs\experiment_setup.png" width="350">

I chose to have a two-layer set up which also complies with Dale’s law since there is a separate excitatory (red) and inhibitory (blue) population.
The independent variable is the wiring motif between the excitatory and inhibitory layer, which is fixed for each experiment. 
  
  - E-Layer: [DiehlAndCookNodes()](https://github.com/BindsNET/bindsnet/blob/master/bindsnet/network/nodes.py#L981)
  
  - I-Layer: [LIFNodes()](https://github.com/BindsNET/bindsnet/blob/master/bindsnet/network/nodes.py#L418)
  
  - fixed E→I, E←I connections
  
  - train E-Layer with STDP
    
  - Data: MNIST encoded with a [PoissonEncoder()](https://github.com/BindsNET/bindsnet/blob/master/bindsnet/encoding/encoders.py#L88).
<br>



---
### Special Requirements 
Install BindsNET with 

``!pip install numpy scipy matplotlib git+https://github.com/BindsNET/bindsnet.git``

(should automatically install torch) 



