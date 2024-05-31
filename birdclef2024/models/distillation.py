import torch.nn as nn

__all__ = [
    "TeacherStudentModel",
]


class TeacherStudentModel(nn.Module):
    """Base class of teacher-student model for knowledge distillation."""

    def __init__(self, teacher: nn.Module, student: nn.Module) -> None:
        super().__init__()

        self.teacher = teacher
        self.student = student
