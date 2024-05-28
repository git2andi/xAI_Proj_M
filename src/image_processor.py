import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

class Visualization:
    @staticmethod
    def plot_images(images, titles, heading, nrows, ncols, save_path=None):
        print("Preparing to plot images...")
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5))
        fig.suptitle(heading, fontsize=16)
        for i, ax in enumerate(axes.flatten()):
            if i < len(images):
                img = images[i]
                if isinstance(img, torch.Tensor):
                    print(f"Original tensor shape: {img.shape}")
                    if img.ndim == 3:
                        img = img.permute(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
                        print(f"Permuted tensor shape: {img.shape}")
                    img = img.numpy()
                elif isinstance(img, np.ndarray):
                    print(f"Numpy array shape: {img.shape}")
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))  # Change shape from (C, H, W) to (H, W, C)
                        print(f"Transposed numpy array shape: {img.shape}")
                else:
                    raise TypeError(f"Unexpected image type: {type(img)}")

                img = (img - img.min()) / (img.max() - img.min())  # Normalize image data
                img = img.astype('float32')  # Convert to a supported dtype
                print(f"Image shape for plotting: {img.shape}, dtype: {img.dtype}")
                ax.imshow(img)
                ax.set_title(titles[i])
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.85])
        
        if save_path:
            print(f"Saving image to {save_path}")
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_tsne(x_embedded, y_train, idx, save_path=None):
        print("Preparing to plot t-SNE...")
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1], hue=y_train, palette="pastel", legend="full")
        plt.scatter(x_embedded[idx, 0], x_embedded[idx, 1], color='red', s=100, edgecolor='black', label=f'Index {idx}')
        plt.annotate(f'Index {idx}', (x_embedded[idx, 0], x_embedded[idx, 1]), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.title('CIFAR10 - t-SNE')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        
        if save_path:
            print(f"Saving t-SNE plot to {save_path}")
            plt.savefig(save_path)
        plt.show()
