from sconce import journals, progress_monitors, rate_controllers

import math
import numpy as np
import tempfile
import torch


class BasicTrainer:
    def __init__(self, *, model, training_data_generator, test_data_generator,
                 optimizer, journal=None, progress_monitor=None,
                 rate_controller=None):
        self.model = model

        self.training_data_generator = training_data_generator

        self.test_data_generator = test_data_generator

        if journal is None:
            journal = journals.DataframeJournal()
        self.journal = journal

        if progress_monitor is None:
            metric_names = {'training_loss': 'loss', 'test_loss': 'val_loss'}
            progress_monitor = progress_monitors.StdoutProgressMonitor(
                    metric_names=metric_names)
        self.progress_monitor = progress_monitor

        if rate_controller is None:
            rate_controller = rate_controllers.CosineRateController(
                    max_learning_rate=1e-4)
        self.rate_controller = rate_controller

        self.train_to_test_ratio = (len(training_data_generator) //
                                    len(test_data_generator))

        self.optimizer = optimizer

        self.checkpoint_filename = None

    def checkpoint(self, filename=None):
        filename = self.save_model_state(filename=filename)
        self.checkpoint_filename = filename
        return filename

    def save_model_state(self, filename=None):
        if filename is None:
            with tempfile.NamedTemporaryFile() as ofile:
                filename = ofile.name
        torch.save(self.model.state_dict(), filename)
        return filename

    def restore(self):
        if self.checkpoint_filename is None:
            raise RuntimeError("You haven't checkpointed this trainer's "
                    "model yet!")
        self.load_model_state(self.checkpoint_filename)

    def load_model_state(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def train(self, *, num_epochs, journal=None, progress_monitor=None,
            rate_controller=None):
        if journal is None:
            journal = self.journal
        if progress_monitor is None:
            progress_monitor = self.progress_monitor
        if rate_controller is None:
            rate_controller = self.rate_controller

        num_steps = math.ceil(num_epochs * len(self.training_data_generator))
        return self._train(num_steps=num_steps,
                journal=journal,
                progress_monitor=progress_monitor,
                rate_controller=rate_controller)

    def _train(self, *, num_steps, rate_controller,
            journal, progress_monitor):
        progress_monitor.start_session(num_steps)
        rate_controller.start_session(num_steps)

        iterations_since_test = self.train_to_test_ratio

        step_data = {}
        for i in range(num_steps):
            new_learning_rate = rate_controller.new_learning_rate(
                    step=i, data=step_data)
            self._update_learning_rate(new_learning_rate)

            training_step_dict = self._do_training_step()

            iterations_since_test += 1
            if iterations_since_test >= self.train_to_test_ratio:
                test_step_dict = self._do_test_step()
                iterations_since_test = 0

            step_data = {'learning_rate': new_learning_rate,
                    **training_step_dict,
                    **test_step_dict}

            journal.record_step(step_data)
            progress_monitor.step(step_data)

        return journal

    def _update_learning_rate(self, new_learning_rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate
        return new_learning_rate

    def _do_training_step(self):
        self.optimizer.zero_grad()

        inputs, targets = self.training_data_generator.next()
        step_dict = self._do_step(inputs, targets, train=True)
        loss = step_dict['loss']

        loss.backward()
        self.optimizer.step()

        return {f'training_{k}': v for k, v in step_dict.items()}

    def _do_step(self, inputs, targets, train):
        run_dict = self._run_model(inputs, targets, train=train)
        loss_dict = self.model.calculate_loss(**run_dict)
        return {**loss_dict, **run_dict}

    def _run_model(self, inputs, targets, train):
        self.model.train(train)
        in_dict = {'inputs': inputs, 'targets': targets}

        out_dict = self.model(**in_dict)
        return {**out_dict, **in_dict}

    def _do_test_step(self):
        inputs, targets = self.test_data_generator.next()
        step_dict = self._do_step(inputs, targets, train=False)
        return {f'test_{k}': v for k, v in step_dict.items()}

    def test(self, *, journal=None, progress_monitor=None):
        if journal is None:
            journal = journals.DataframeJournal()

        if progress_monitor is None:
            metric_names = {'test_loss': 'loss'}
            progress_monitor = progress_monitors.StdoutProgressMonitor(
                    metric_names=metric_names)

        num_steps = len(self.training_data_generator)
        progress_monitor.start_session(num_steps)

        for i in range(num_steps):
            step_data = self._do_test_step()

            journal.record_step(step_data)
            progress_monitor.step(step_data)

        return journal

    def multi_train(self, *, num_cycles, cycle_len=1,
            cycle_multiplier=2.0, **kwargs):
        for cycle in range(1, num_cycles + 1):
            scale_factor = cycle * cycle_multiplier
            num_epochs = cycle_len * scale_factor
            self.train(num_epochs=num_epochs, **kwargs)

    def survey_learning_rate(self, *, num_epochs=1.0,
            min_learning_rate=1e-12,
            max_learning_rate=10,
            journal=None,
            progress_monitor=None,
            rate_controller_class=rate_controllers.ExponentialRateController):
        if journal is None:
            journal = journals.DataframeJournal()

        if progress_monitor is None:
            metric_names = {'training_loss': 'loss'}
            progress_monitor = progress_monitors.StdoutProgressMonitor(
                    metric_names=metric_names)

        filename = self.save_model_state()

        rate_controller = rate_controller_class(
                min_learning_rate=min_learning_rate,
                max_learning_rate=max_learning_rate)
        self.train(num_epochs=num_epochs,
                journal=journal,
                progress_monitor=progress_monitor,
                rate_controller=rate_controller)

        self.load_model_state(filename)

        return journal

    @property
    def num_trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad,
                self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
