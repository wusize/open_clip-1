import torch
import argparse
from open_clip.factory import create_model
@torch.no_grad()
def student_teacher_ensemble(student, teacher, alpha=0.5):
    for param_student, param_teacher in zip(
            student.parameters(), teacher.parameters()):
        param_student.data = param_student.data * alpha + param_teacher.data * (1.0 - alpha)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract and save the CLIP visual weights')
    parser.add_argument('--model', default='ViT-B-16', type=str)
    parser.add_argument('--cache-dir', default='./checkpoints', type=str)
    parser.add_argument('--pretrained', default="", type=str)
    parser.add_argument('--output', default="", type=str)
    parser.add_argument('--alpha', default=0.5, type=float)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    student = create_model(model_name=args.model, pretrained=args.pretrained)
    teacher = create_model(model_name=args.model, pretrained="openai", cache_dir=args.cache_dir)
    student_teacher_ensemble(student=student, teacher=teacher, alpha=args.alpha)
    checkpoint_dict = {
        "state_dict": student.state_dict(),
    }

    torch.save(checkpoint_dict, args.output)
