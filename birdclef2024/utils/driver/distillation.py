import os
from typing import Optional

import torch
from audyn.criterion import BaseCriterionWrapper
from audyn.utils.clip_grad import GradClipper
from audyn.utils.data import BaseDataLoaders
from audyn.utils.driver import BaseTrainer
from audyn.utils.logging import get_logger
from audyn.utils.model import unwrap
from audyn.utils.tensorboard import get_summary_writer
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ...models.distillation import TeacherStudentModel


class StudentTrainer(BaseTrainer):
    def __init__(
        self,
        loaders: BaseDataLoaders,
        model: TeacherStudentModel,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        grad_clipper: Optional[GradClipper] = None,
        criterion: BaseCriterionWrapper = None,
        config: DictConfig = None,
    ) -> None:
        super().__init__(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            grad_clipper=grad_clipper,
            criterion=criterion,
            config=config,
        )

    def _reset(self, config: DictConfig) -> None:
        self.set_system(config=config.system)

        self.scaler = GradScaler(enabled=self.enable_amp)

        epochs = config.train.steps.epochs
        iterations = config.train.steps.iterations

        assert (epochs is not None and iterations is None) or (
            epochs is None and iterations is not None
        ), "Define either of config.train.epochs and config.train.iterations."

        if epochs is None:
            self.epochs = (iterations - 1) // len(self.loaders.train) + 1
            self.iterations = iterations
        else:
            self.epochs = epochs
            self.iterations = len(self.loaders.train) * epochs

        self.exp_dir = config.train.output.exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)

        # Set git commit hash
        self.set_commit_hash()

        # Set loggder
        self.logger = get_logger(self.__class__.__name__, is_distributed=self.is_distributed)

        # Set tensorboard writer
        self.writer = get_summary_writer(
            log_dir=config.train.output.tensorboard_dir, is_distributed=self.is_distributed
        )

        # Display config and model architecture after logger instantiation
        self.logger.info(OmegaConf.to_yaml(self.config))
        self.display_model(display_num_parameters=True)

        self.iteration_idx = 0
        self.best_loss = float("inf")
        self.epoch_idx = 0

        if config.train.checkpoint.teacher:
            teacher_checkpoint = config.train.checkpoint.teacher
            self.logger.info(
                f"Load weights of pretrained teacher model from: {teacher_checkpoint}."
            )
            self.load_teacher_checkpoint(teacher_checkpoint)

        if config.train.checkpoint.student:
            student_checkpoint = config.train.checkpoint.student
            self.logger.info(
                f"Load weights of pretrained student model from: {student_checkpoint}."
            )
            self.load_student_checkpoint(student_checkpoint)

        if config.train.resume.continue_from:
            continue_from = config.train.resume.continue_from
            self.logger.info(f"Resume training from: {continue_from}.")
            self.load_checkpoint(continue_from)

    def load_teacher_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        unwrapped_model: TeacherStudentModel = self.unwrapped_model
        teacher = unwrap(unwrapped_model.teacher)
        teacher.load_state_dict(state_dict["model"])

    def load_student_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        unwrapped_model: TeacherStudentModel = self.unwrapped_model
        student = unwrap(unwrapped_model.student)
        student.load_state_dict(state_dict["model"])
