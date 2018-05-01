from collections import OrderedDict
from sconce import monitors, rate_controllers

import copy
import math
import numpy as np
import tempfile
import torch


class Trainer:
    """
    A Class that is used to train pytorch models.
    It defines the training loop and orchestrates the various other sconce
    objects (:py:class:`~sconce.data_generators.base.DataGenerator`,
    :py:class:`~sconce.monitors.base.Monitor`,
    :py:class:`~sconce.rate_controllers.base.RateController`, ect).

    Keyword Arguments:
        model (:py:class:`torch.nn.Module`): the torch model to be trained.  See :py:mod:`sconce.models` for examples.
        training_data_generator (:py:class:`~sconce.data_generators.base.DataGenerator`): yields training `inputs` and
            `targets`.
        test_data_generator (:py:class:`~sconce.data_generators.base.DataGenerator`): yields test `inputs` and
            `targets`.  These are never used for back-propagation.
        optimizer (:py:class:`torch.optim.Optimizer`): the torch optimizer that updates model parameters during
            training.
        monitor (:py:class:`~sconce.monitors.base.Monitor`, optional): the sconce monitor that records data during
            training.  This data can be sent to external systems during training or kept until training completes
            allowing you to analyze training or make plots. If ``None``, a composite monitor consisting of a
            :py:class:`~sconce.monitors.stdout_monitor.StdoutMonitor` and a
            :py:class:`~sconce.monitors.dataframe_monitor.DataframeMonitor` will be created for you and used.
        rate_controller (:py:class:`~sconce.rate_controllers.base.RateController`, optional): a sconce rate_monitor
            that adjusts the optimizer's learning rate during training. If ``None``, a
            :py:class:`~sconce.rate_controllers.cosine_rate_controller.CosineRateController` with
            max_learning_rate=1e-4 will be created for you and used.
    """
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
        self._reset_cache()

    def _reset_cache(self):
        self._cache_data_generator = None
        self._inputs = None
        self._targets = None
        self._outputs = None

    def checkpoint(self, filename=None):
        """
        Save model state and retain filename for a later call to :py:meth:`~sconce.trainer.Trainer.restore`.

        Arguments:
            filename (path, optional): the filename to save the model state to.
        """
        filename = self.save_model_state(filename=filename)
        self.checkpoint_filename = filename
        return filename

    def save_model_state(self, filename=None):
        """
        Save model state to a file.

        Arguments:
            filename (path, optional): the filename to save the model state to.
                If ``None``, a system dependent temporary location will be chosen.

        Returns:
            filename (path): the passed in filename, or the temporary filename chosen if ``None`` was passed in.
        """
        if filename is None:
            with tempfile.NamedTemporaryFile() as ofile:
                filename = ofile.name
        torch.save(self.model.state_dict(), filename)
        return filename

    def restore(self):
        """
        Restore model to previously checkpointed state.  See also :py:meth:`~sconce.trainer.Trainer.checkpoint`.
        """
        if self.checkpoint_filename is None:
            raise RuntimeError("You haven't checkpointed this trainer's "
                    "model yet!")
        self.load_model_state(self.checkpoint_filename)

    def load_model_state(self, filename):
        """
        Restore model state frome a file.

        Arguments:
            filename (path): the filename to where the model's state was saved.
        """
        self.model.load_state_dict(torch.load(filename))

    def train(self, *, num_epochs, monitor=None,
            rate_controller=None, test_to_train_ratio=None,
            batch_multiplier=1):
        """
        Train the model for a given number of epochs.

        Arguments:
            num_epochs (float): the number of epochs to train the model for.
            monitor (:py:class:`~sconce.monitors.base.Monitor`, optional): a monitor to use for this training session.
                If ``None``, then self.monitor will be used.
            rate_controller (:py:class:`~sconce.rate_controllers.base.RateController, optional): a rate_controller to
                use for this training session.  If ``None``, then self.rate_controller will be used.
            test_to_train_ratio (float, optional): [0.0, 1.0] determines how often (relative to training samples) that
                test samples are run through the model during training.  If ``None``, then the relative size of the
                training and test datasets is used.  For example, for MNIST with 60,000 training samples and 10,000 test
                samples, the value would be 1/6th.
            batch_multiplier (int, optional): [1, inf) determines how often parameter updates will occur during
                training.  If greater than 1, this simulates large batch sizes without increasing memory usage.  For
                example, if the batch size were 100 and batch_multipler=10, the effective batch size would be 1,000, but
                the memory usage would be for a batch size of 100.

        Returns:
            monitor (:py:class:`~sconce.monitors.base.Monitor`): the monitor used during training.
        """
        assert batch_multiplier > 0
        assert int(batch_multiplier) == batch_multiplier

        if monitor is None:
            monitor = self.monitor
        if rate_controller is None:
            rate_controller = self.rate_controller
        if test_to_train_ratio is None:
            test_to_train_ratio = self.test_to_train_ratio

        num_steps = math.ceil(num_epochs * len(self.training_data_generator))
        num_steps //= batch_multiplier

        return self._train(num_steps=num_steps,
                monitor=monitor,
                rate_controller=rate_controller,
                test_to_train_ratio=test_to_train_ratio,
                batch_multiplier=batch_multiplier)

    def _train(self, *, num_steps, rate_controller, monitor,
            test_to_train_ratio, batch_multiplier):
        self._reset_cache()
        monitor.start_session(num_steps)
        rate_controller.start_session(num_steps)

        iterations_since_test = 0

        monitor_data = {}
        for step in range(1, num_steps + 1):
            new_learning_rate = rate_controller.new_learning_rate(
                    step=step, data=monitor_data)
            if new_learning_rate is None:
                break

            self._update_learning_rate(new_learning_rate)

            self.optimizer.zero_grad()
            for i in range(1, batch_multiplier + 1):
                inputs, targets = self.training_data_generator.next()
                step_dict = self._do_step(inputs, targets, train=True)

                loss = step_dict['loss'] / batch_multiplier
                loss.backward()

                training_step_dict = {f'training_{k}': v
                        for k, v in step_dict.items()}

                iterations_since_test += 1
                if (1 / iterations_since_test) <= test_to_train_ratio:
                    test_step_dict = self._do_test_step()
                    iterations_since_test = 0

                    monitor_data = {'learning_rate': new_learning_rate,
                            **training_step_dict,
                            **test_step_dict}
                else:
                    monitor_data = {'learning_rate': new_learning_rate,
                            **training_step_dict}

                fraction = i / batch_multiplier
                monitor.write(data=monitor_data, step=step - 1 + fraction)

            self.optimizer.step()

        monitor.end_session()

        return monitor

    def _update_learning_rate(self, new_learning_rate):
        param_groups = self.optimizer.param_groups
        if isinstance(new_learning_rate, OrderedDict):
            assert len(new_learning_rate.values()) == len(param_groups),\
                    "Expected the same number of learning rates as param_groups defined by the optimizer "\
                    f"{len(param_groups)}, but found {len(new_learning_rate.values())} learning rates instead."
            for param_group, lr in zip(param_groups, new_learning_rate.values()):
                param_group['lr'] = lr
        else:
            for param_group in param_groups:
                param_group['lr'] = new_learning_rate

        return new_learning_rate

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

    def _run_model_on_generator(self, data_generator,
            cache_results=True):
        if self._cache_data_generator is data_generator:
            return {'inputs': self._inputs,
                    'targets': self._targets,
                    'outputs': self._outputs}

        inputs = []
        targets = []
        outputs = []

        data_generator.reset()
        for x in range(len(data_generator)):
            i, t = data_generator.next()
            out_dict = self._run_model(i, t, train=False)

            inputs.append(i.cpu().data.numpy())
            targets.append(t.cpu().data.numpy())
            outputs.append(out_dict['outputs'].cpu().data.numpy())

        inputs = np.concatenate(inputs)
        targets = np.concatenate(targets)
        outputs = np.concatenate(outputs)

        if cache_results:
            self._cache_data_generator = data_generator
            self._inputs = inputs
            self._targets = targets
            self._outputs = outputs

        return {'inputs': inputs,
                'targets': targets,
                'outputs': outputs}

    def _do_test_step(self):
        inputs, targets = self.test_data_generator.next()
        step_dict = self._do_step(inputs, targets, train=False)
        return {f'test_{k}': v for k, v in step_dict.items()}

    def test(self, *, monitor=None):
        """
        Run all samples of self.test_data_generator through the model in test (inference) mode.

        Arguments:
            monitor (:py:class:`~sconce.monitors.base.Monitor`, optional): the sconce monitor that records data during
                this testing.  If ``None``, a composite monitor consisting of a
                :py:class:`~sconce.monitors.stdout_monitor.StdoutMonitor` and a
                :py:class:`~sconce.monitors.dataframe_monitor.DataframeMonitor` will be created for you and used.

        Returns:
            monitor (:py:class:`~sconce.monitors.base.Monitor`): the monitor used during this testing.
        """
        if monitor is None:
            metric_names = {'test_loss': 'loss'}
            stdout_monitor = monitors.StdoutMonitor(metric_names=metric_names)
            monitor = monitors.DataframeMonitor() + stdout_monitor

        num_steps = len(self.training_data_generator)
        monitor.start_session(num_steps)

        for step in range(1, num_steps + 1):
            step_data = self._do_test_step()

            monitor.write(data=step_data, step=step)
        monitor.end_session()

        return monitor

    def multi_train(self, *, num_cycles, cycle_length=1,
            cycle_multiplier=2.0, **kwargs):
        """
        Runs multiple training sessions one after another.

        Arguments:
            num_cycles (int): [1, inf) the number of cycles to train for.
            cycle_length (float): (0.0, inf) the length (in epochs) of the first cycle.
            cycle_multiplier (float): (0.0, inf) a factor used to determine the length of a cycle.  The length of a
                cycle is equal to the length of the previous cycle (or ``cycle_length`` if it is the first cycle)
                multiplied by ``cycle_multiplier``.

        Keyword Arguments:
            **kwargs: are passed to the underlying :py:meth:`~sconce.trainer.Trainer.train` method.
        """
        this_cycle_length = cycle_length
        for i in range(num_cycles):
            self.train(num_epochs=this_cycle_length, **kwargs)
            this_cycle_length *= cycle_multiplier

    def survey_learning_rate(self, *, num_epochs=1.0,
            min_learning_rate=1e-12,
            max_learning_rate=10,
            monitor=None,
            batch_multiplier=1,
            rate_controller_class=rate_controllers.ExponentialRateController,
            stop_factor=10,
            **rate_controller_kwargs):
        """
        Checkpoints a model, then runs a learning rate survey, before restoring the model back.

        Keyword Arguments:
            num_epochs (float, optional): (0.0, inf) the number of epochs to train the model for.
            min_learning_rate (float, optional): (0.0, inf) the minimum learning rate used in the survey.
            max_learning_rate (float, optional): (0.0, inf) the maximum learning rate used in the survey.
            monitor (:py:class:`~sconce.monitors.base.Monitor`, optional): the sconce monitor that records data during
                the learning rate survey.  If ``None``, a composite monitor consisting of a
                :py:class:`~sconce.monitors.stdout_monitor.StdoutMonitor` and a
                :py:class:`~sconce.monitors.dataframe_monitor.DataframeMonitor` will be created for you and used.
            batch_multiplier (int, optional): [1, inf) determines how often parameter updates will occur during
                training.  If greater than 1, this simulates large batch sizes without increasing memory usage.  For
                example, if the batch size were 100 and batch_multipler=10, the effective batch size would be 1,000, but
                the memory usage would be for a batch size of 100.
            rate_controller_class (class, optional): the subclass of
                :py:class:`~sconce.rate_controllers.base.RateController` to be used during this learning rate survey.
                In practice, only
                :py:class:`~sconce.rate_controllers.exponential_rate_controller.ExponentialRateController` and
                :py:class:`~sconce.rate_controllers.linear_rate_controller.LinearRateController` are typically used.
            stop_factor (float): (1.0, inf) determines early stopping.  If the `training loss` rises by more than
                this factor from it's minimum value, the survey will stop.
            **rate_controller_kwargs: passed to the constructor of the ``rate_controller_class`` along with
                ``min_learning_rate``, ``max_learning_rate``, and ``stop_factor``.

        Returns:
            monitor (:py:class:`~sconce.monitors.base.Monitor`): the monitor used during this learning rate survey.
        """
        if monitor is None:
            metric_names = {'training_loss': 'loss'}
            stdout_monitor = monitors.StdoutMonitor(metric_names=metric_names)
            monitor = monitors.DataframeMonitor() + stdout_monitor

        orig_model_state_dict = copy.deepcopy(self.model.state_dict())
        orig_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())

        rate_controller = rate_controller_class(
                min_learning_rate=min_learning_rate,
                max_learning_rate=max_learning_rate,
                stop_factor=stop_factor,
                **rate_controller_kwargs)
        self.train(num_epochs=num_epochs,
                monitor=monitor,
                rate_controller=rate_controller,
                test_to_train_ratio=0,
                batch_multiplier=batch_multiplier)

        self.model.load_state_dict(orig_model_state_dict)
        self.optimizer.load_state_dict(orig_optimizer_state_dict)

        return monitor

    @property
    def num_trainable_parameters(self):
        """
        The number of trainable parameters that the models has.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
