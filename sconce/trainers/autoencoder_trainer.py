from abc import ABC
from sconce.trainer import Trainer

from matplotlib import pyplot as plt


__all__ = ['AutoencoderMixin', 'AutoencoderTrainer']


class AutoencoderMixin(ABC):
    def plot_input_output_pairs(self, title='A Sampling of Autoencoder Results',
        num_cols=10, figsize=(15, 3.2)):
        inputs, targets = self.test_data_generator.next()
        out_dict = self._run_model(inputs, targets, train=True)
        outputs = out_dict['outputs']

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20)

        for i in range(num_cols):
            input_image = inputs.data.cpu()[i][0]
            output_image = outputs.view_as(inputs).data.cpu()[i][0]

            ax = fig.add_subplot(2, num_cols, i + 1)
            ax.imshow(input_image, cmap='gray')
            if i == 0:
                ax.set_ylabel('Input')
            else:
                ax.axis('off')

            ax = fig.add_subplot(2, num_cols, num_cols + i + 1)
            ax.imshow(output_image, cmap='gray')
            if i == 0:
                ax.set_ylabel('Output')
            else:
                ax.axis('off')
        return fig

    def plot_latent_space(self, title="Latent Representation", figsize=(8, 8)):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20)

        self.model.train(False)
        self.test_data_generator.reset()
        for i in range(len(self.test_data_generator)):
            inputs, targets = self.test_data_generator.next()
            x_latent = self.model.encode(inputs=inputs, targets=targets)

            x_latent_numpy = x_latent.cpu().data.numpy()
            plt.scatter(x=x_latent_numpy.T[0], y=x_latent_numpy.T[1],
                        c=targets.cpu().data.numpy(), alpha=0.4)
        plt.colorbar()
        return fig


class AutoencoderTrainer(Trainer, AutoencoderMixin):
    pass
