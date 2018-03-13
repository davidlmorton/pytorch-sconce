from .base import MNISTTest
from sconce.data_generator import DataGenerator
from sconce.rate_controllers import CosineRateController
from sconce.trainers import AutoencoderTrainer
from sconce.models import BasicAutoencoder
from torch import optim

import torch


class TestBasicAutoencoder(MNISTTest):
    num_training_samples = 10_000
    num_test_samples = 1_000

    def test_full_run(self):
        RANDOM_SEED = 1
        torch.manual_seed(RANDOM_SEED)

        model = BasicAutoencoder(image_height=28, image_width=28,
                hidden_size=200, latent_size=100)
        training_generator = DataGenerator(self.training_data_loader)
        test_generator = DataGenerator(self.test_data_loader)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
            model.cuda()
            training_generator.cuda()
            test_generator.cuda()

        optimizer = optim.SGD(model.parameters(), lr=1e-4,
                momentum=0.9, weight_decay=1e-6)

        trainer = AutoencoderTrainer(model=model, optimizer=optimizer,
                training_data_generator=training_generator,
                test_data_generator=test_generator)

        survey_monitor = trainer.survey_learning_rate(num_epochs=0.1,
                min_learning_rate=1e-1, max_learning_rate=1e3)
        survey_monitor.dataframe_monitor.plot_learning_rate_survey()

        trainer.plot_input_output_pairs()
        trainer.plot_latent_space()

        rate_controller = CosineRateController(max_learning_rate=2)
        trainer.train(num_epochs=1, rate_controller=rate_controller)

        trainer.multi_train(num_cycles=2, rate_controller=rate_controller)
        trainer.monitor.dataframe_monitor.plot()

        test_monitor = trainer.test()
        test_loss = test_monitor.dataframe_monitor.df['test_loss'].mean()
        self.assertTrue(test_loss < 0.21)
