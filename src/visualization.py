import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

class Visualization:
    @staticmethod
    def plot_images(images, titles, heading, nrows, ncols, save_path=None):
        """Plot a grid of images with titles."""
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5))
        fig.suptitle(heading, fontsize=16)
        for i, ax in enumerate(axes.flatten()):
            if i < len(images):
                img = images[i]
                if isinstance(img, torch.Tensor):
                    if img.ndim == 3:
                        img = img.permute(1, 2, 0)
                    img = img.numpy()
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[0] == 1:  # Grayscale image
                        img = np.squeeze(img, axis=0)
                    elif img.ndim == 3 and img.shape[0] == 3:  # RGB image
                        img = np.transpose(img, (1, 2, 0))
                else:
                    raise TypeError(f"Unexpected image type: {type(img)}")

                img = (img - img.min()) / (img.max() - img.min())  # Normalize image data
                img = img.astype('float32')  # Convert to a supported dtype
                ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                ax.set_title(titles[i])
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.85])
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_tsne(x_embedded, y_train, idx=None, save_path=None):
        """Plot t-SNE visualization."""
        #plt.figure(figsize=(10, 8))
        sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=y_train, palette="pastel") #, legend="full"
        
        if idx is not None:
            plt.scatter(x_embedded[idx, 0], x_embedded[idx, 1], color='red', s=100, edgecolor='black', label=f'Index {idx}')
            plt.annotate(f'Index {idx}', (x_embedded[idx, 0], x_embedded[idx, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
        
        plt.title('t-SNE Visualization')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        if save_path:
            plt.savefig(save_path)
        plt.show()
