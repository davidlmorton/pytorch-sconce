from sconce import monitors, rate_controllers

import math
import numpy as np
import tempfile
import torch


class Trainer:
    def __init__(self, *, model, training_data_generator, test_data_generator,
                 optimizer, monitor=None, rate_controller=None):
        self.model = model

        self.training_data_generator = training_data_generator

        self.test_data_generator = test_data_generator

        if monitor is None:
            metric_names = {'training_loss': 'loss', 'test_loss': 'val_loss'}
            stdout_monitor = monitors.StdoutMonitor(metric_names=metric_names)
            monitor = monitors.DataframeMonitor() + stdout_monitor
        self.monitor = monitor

        if rate_controller is None:
            rate_controller = rate_controllers.CosineRateController(
                    max_learning_rate=1e-4)
        self.rate_controller = rate_controller

        self.test_to_train_ratio = (len(test_data_generator) /
                                    len(training_data_generator))

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

    def train(self, *, num_epochs, monitor=None,
            rate_controller=None, test_to_train_ratio=None):
        if monitor is None:
            monitor = self.monitor
        if rate_controller is None:
            rate_controller = self.rate_controller
        if test_to_train_ratio is None:
            test_to_train_ratio = self.test_to_train_ratio

        num_steps = math.ceil(num_epochs * len(self.training_data_generator))
        return self._train(num_steps=num_steps,
                monitor=monitor,
                rate_controller=rate_controller,
                test_to_train_ratio=test_to_train_ratio)

    def _train(self, *, num_steps, rate_controller, monitor,
            test_to_train_ratio):
        monitor.start_session(num_steps)
        rate_controller.start_session(num_steps)

        iterations_since_test = 0

        step_data = {}
        for i in range(num_steps):
            new_learning_rate = rate_controller.new_learning_rate(
                    step=i, data=step_data)
            if new_learning_rate is None:
                break

            self._update_learning_rate(new_learning_rate)

            training_step_dict = self._do_training_step()

            iterations_since_test += 1
            if (1 / iterations_since_test) <= test_to_train_ratio:
                test_step_dict = self._do_test_step()
                iterations_since_test = 0

                step_data = {'learning_rate': new_learning_rate,
                        **training_step_dict,
                        **test_step_dict}
            else:
                step_data = {'learning_rate': new_learning_rate,
                        **training_step_dict}

            monitor.step(step_data)
        monitor.end_session()

        return monitor

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

        if hasattr(self.model, 'calculate_metrics'):
            metrics_dict = self.model.calculate_metrics(**run_dict,
                    **loss_dict)
            return {**metrics_dict, **loss_dict, **run_dict}
        else:
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

    def test(self, *, monitor=None):
        if monitor is None:
            metric_names = {'test_loss': 'loss'}
            stdout_monitor = monitors.StdoutMonitor(metric_names=metric_names)
            monitor = monitors.DataframeMonitor() + stdout_monitor

        num_steps = len(self.training_data_generator)
        monitor.start_session(num_steps)

        for i in range(num_steps):
            step_data = self._do_test_step()

            monitor.step(step_data)
        monitor.end_session()

        return monitor

    def multi_train(self, *, num_cycles, cycle_len=1,
            cycle_multiplier=2.0, **kwargs):
        for cycle in range(1, num_cycles + 1):
            scale_factor = cycle * cycle_multiplier
            num_epochs = cycle_len * scale_factor
            self.train(num_epochs=num_epochs, **kwargs)

    def survey_learning_rate(self, *, num_epochs=1.0,
            min_learning_rate=1e-12,
            max_learning_rate=10,
            monitor=None,
            rate_controller_class=rate_controllers.ExponentialRateController,
            **rate_controller_kwargs):

        if monitor is None:
            metric_names = {'training_loss': 'loss'}
            stdout_monitor = monitors.StdoutMonitor(metric_names=metric_names)
            monitor = monitors.DataframeMonitor() + stdout_monitor

        filename = self.save_model_state()

        rate_controller = rate_controller_class(
                min_learning_rate=min_learning_rate,
                max_learning_rate=max_learning_rate,
                **rate_controller_kwargs)
        self.train(num_epochs=num_epochs,
                monitor=monitor,
                rate_controller=rate_controller,
                test_to_train_ratio=0)

        self.load_model_state(filename)

        return monitor

    @property
    def num_trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad,
                self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
