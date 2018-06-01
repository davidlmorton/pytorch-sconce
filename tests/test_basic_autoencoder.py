from sconce.data_generators import SingleClassImageDataGenerator
from sconce.schedules import Cosine
from sconce.trainers import AutoencoderTrainer
from sconce.models import BasicAutoencoder
from torch import optim

import torch
import unittest


class TestBasicAutoencoder(unittest.TestCase):
    def test_full_run(self):
        model = BasicAutoencoder(image_height=28, image_width=28,
                hidden_size=200, latent_size=100)

        training_generator = SingleClassImageDataGenerator.from_torchvision(fraction=1 / 6)
        test_generator = SingleClassImageDataGenerator.from_torchvision(train=False, fraction=0.1)

        if torch.cuda.is_available():
            model.cuda()
            training_generator.cuda()
            test_generator.cuda()

        model.set_optimizer(optim.SGD, lr=1e-4, momentum=0.9, weight_decay=1e-6)

        trainer = AutoencoderTrainer(model=model,
                training_data_generator=training_generator,
                test_data_generator=test_generator)

        survey_monitor = trainer.survey_learning_rate(num_epochs=0.1,
                min_learning_rate=1e-1, max_learning_rate=1e3)
        survey_monitor.dataframe_monitor.plot_learning_rate_survey()

        trainer.plot_input_output_pairs()
        trainer.plot_latent_space()

        model.set_schedule('learning_rate', Cosine(initial_value=2, final_value=2 / 50))
        trainer.train(num_epochs=1)

        trainer.multi_train(num_cycles=4)
        trainer.monitor.dataframe_monitor.plot()

        test_monitor = trainer.test()
        test_loss = test_monitor.dataframe_monitor.df['test_loss'].mean()
        self.assertLess(test_loss, 0.21)
