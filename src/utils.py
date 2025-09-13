import numpy as np
import matplotlib.pyplot as plt

def draw_weights(weights, step, im=None, n_rows=10, n_cols=10, figsize=(8,8)):

    w = weights.detach().clone().cpu().numpy().T
    pxl_x = pxl_y = int((w.shape[1])**(1/2))
    n_units,_ = w.shape

    reshaped_w = np.zeros((pxl_y*n_rows,pxl_x*n_cols))

    unit = 0
    for y in range(n_rows):
        for x in range(n_cols):
            reshaped_w[y*pxl_y:(y+1)*pxl_y, x*pxl_x:(x+1)*pxl_x]=w[unit,:].reshape(pxl_y,pxl_x)
            unit += 1
        
    if not im:
            fig, ax = plt.subplots(figsize=figsize)

            im = plt.imshow(reshaped_w, cmap='hot_r', vmin=np.min(reshaped_w), vmax=np.max(reshaped_w))
            fig.colorbar(im, ticks=[np.amin(reshaped_w), 0, np.amax(reshaped_w)])

            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_aspect("auto")
            plt.title(f"Weights at step: {step}", pad=20)

            fig.tight_layout()
    else:
        im.set_data(reshaped_w)
        wmin = weights.min().item()
        wmax = weights.max().item()
        im.set_clim(wmin, wmax)
       # fig.colorbar(im, ticks=[np.amin(reshaped_w), 0, np.amax(reshaped_w)])
        plt.title(f"Weights at step: {step}", pad=20)

    # show 
    plt.draw()
    return im


def plot_spikes(spikes, im=None):
    
    spks = spikes.detach().clone().cpu().numpy().T

    print(spks)
    if not im:
        fig, ax = plt.subplots(figsize=(5,5))

        im = plt.imshow(spks, cmap='bwr')

        fig.tight_layout()

    else:
         im.set_data(spks)
    plt.draw()
    return im