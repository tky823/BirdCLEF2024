from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
    "TeacherStudentModel",
]


class TeacherStudentModel(nn.Module):
    """Base class of teacher-student model for knowledge distillation."""

    def __init__(
        self, teacher: nn.Module, student: nn.Module, train_teacher: bool = False
    ) -> None:
        super().__init__()

        self.teacher = teacher
        self.student = student

        self.train_teacher = train_teacher

    def train(self, mode: bool = True) -> "TeacherStudentModel":
        if self.train_teacher:
            self.teacher.train(mode=mode)
        else:
            self.teacher.eval()

        self.student.train(mode=mode)

        return self


class SemiSupervisedTeacherStudentModel(TeacherStudentModel):
    """Teacher-student model for semisupervised training."""

    def forward(
        self, labeled_input: torch.Tensor, unlabeled_input: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            unlabeled_teacher_output = self.teacher(unlabeled_input)

        labeled_student_output = self.student(labeled_input)
        unlabeled_student_output = self.student(unlabeled_input)

        teacher_output = (unlabeled_teacher_output,)
        student_output = (labeled_student_output, unlabeled_student_output)
        output = (teacher_output, student_output)

        return output

    @torch.no_grad()
    def inference(self, input: torch.Tensor) -> torch.Tensor:
        output = self.student(input)

        return output
