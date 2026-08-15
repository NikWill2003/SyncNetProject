from __future__ import annotations

from typing import Iterator, TYPE_CHECKING
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from ..contracts import SortOfClevrBatch

if TYPE_CHECKING:
    from ....core.config import Config

def _move_tensor_dict(batch: dict[str, torch.Tensor], device: str):
    
    non_blocking = str(device).startswith('cuda')
    return {key: tensor.to(device, non_blocking=non_blocking) for key, tensor in batch.items()}
        

def _load_sort_of_clevr(path: str):
    with np.load(path) as data:
            
        # dataset might be in float32 or uint8
        images = torch.from_numpy(data['images'])   # (n_scenes, H, W, C)
        if images.dtype != torch.uint8:
            images = images.float()
            if float(images.max()) > 1.0:           # defensive: [0, 255] floats
                images = images / 255.0
        images = images.permute((0, 3, 1, 2)).contiguous()

        # (n_scenes, nb_questions, q_dim)
        ternary_questions = torch.from_numpy(data['ternary_questions']).long() 
        
        # (n_scenes, nb_questions) scalar labels
        ternary_answers = torch.from_numpy(data['ternary_answers']).long() 
        
        # (n_scenes, nb_questions, q_dim)
        binary_questions = torch.from_numpy(data['binary_questions']).long() 
        
        # (n_scenes, nb_questions) scalar labels
        binary_answers = torch.from_numpy(data['binary_answers']).long() 
        
        # (n_scenes, nb_questions, q_dim)
        nonrel_questions = torch.from_numpy(data['nonrel_questions']).long() 

        # (n_scenes, nb_questions) scalar labels 
        nonrel_answers = torch.from_numpy(data['nonrel_answers']).long() 

        n_scenes = images.size(0)
        nb_questions = ternary_questions.size(1)
        total_nb_questions = 3*nb_questions
        q_dim = ternary_questions.size(-1)

        questions = torch.concatenate(
            (ternary_questions, binary_questions, nonrel_questions), dim=-2
            ).view(-1, q_dim) # (n_scenes, 3*nb_questions, q_dim) -> (n_scenes*3*nb_questions, q_dim)
        
        answers = torch.concatenate(
            (ternary_answers, binary_answers, nonrel_answers), dim=-1
            ).flatten() # (n_scenes, 3*nb_questions) -> (n_scenes*3*nb_questions)
        
    return (
        {'images': images, 'questions' :questions, 'answers': answers}, 
        n_scenes, total_nb_questions
        )


def _to_float_images(images: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [0, 1]; float passes through."""
    
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    return images


class SortOfClevrDataset(Dataset):

    def __init__(self, path: str) -> None:
        super().__init__()

        dataset, n_scenes, total_nb_questions = _load_sort_of_clevr(path)
        
        self.images = dataset['images']
        self.questions = dataset['questions']
        self.answers = dataset['answers']
        self.n_scenes = n_scenes
        self.total_nb_questions = total_nb_questions
    
    def __len__(self):
        return self.n_scenes * self.total_nb_questions
    
    def __getitem__(self, idx: int) -> SortOfClevrBatch:

        image_idx = idx // self.total_nb_questions
        return {
            'images': _to_float_images(self.images[image_idx]),
            'questions': self.questions[idx],
            'answers': self.answers[idx],
        }

class SortOfClevrOnDeviceLoader:

    def __init__(
            self,
            path: str,
            batch_size: int,
            device: str,
            shuffle: bool = False,
            ) -> None:

        dataset, n_scenes, total_nb_questions = _load_sort_of_clevr(path)
        self.dataset = _move_tensor_dict(dataset, device)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.total_nb_questions = total_nb_questions
        self.N = n_scenes * total_nb_questions
        assert self.dataset['questions'].shape[0] == self.N, (
            'question/scene count mismatch: '
            f"{self.dataset['questions'].shape[0]} != {self.N}")

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[SortOfClevrBatch]:
        if self.shuffle:
            order = torch.randperm(self.N, device=self.device)
        else:
            order = torch.arange(self.N, device=self.device)

        for i in range(0, self.N, self.batch_size):
            idx = order[i: i + self.batch_size]
            image_idx = idx // self.total_nb_questions
            yield {
                'images': _to_float_images(
                    self.dataset['images'][image_idx]),
                'questions': self.dataset['questions'][idx],
                'answers': self.dataset['answers'][idx],
            }

def build_dataloaders(
        cfg: Config, device: str
        ) -> tuple[DataLoader | SortOfClevrOnDeviceLoader, ...]:
    
    data_dir_path = Path(cfg.dataset.root) / cfg.dataset.dir
    
    train_path = data_dir_path / 'train.npz'
    val_path = data_dir_path / 'val.npz'
    test_path = data_dir_path / 'test.npz'

    if not all(path.exists() for path in [train_path, val_path, test_path]):
        from .generator import prepare_sort_of_clevr
        prepare_sort_of_clevr(cfg.dataset) # type: ignore
    
    if cfg.train.loader_mode == 'gpu_cached':
        return (
            SortOfClevrOnDeviceLoader(
                str(train_path),
                batch_size=cfg.train.train_bs,
                device=device,
                shuffle=True
                ),
            SortOfClevrOnDeviceLoader(
                str(val_path),
                batch_size=cfg.train.val_bs,
                device=device,
                shuffle=False
                ),
            SortOfClevrOnDeviceLoader(
                str(test_path),
                batch_size=cfg.train.val_bs,
                device=device,
                shuffle=False
            )
        )
    elif cfg.train.loader_mode == 'dataloader':
        return (
            DataLoader(
                SortOfClevrDataset(str(train_path)),
                batch_size=cfg.train.train_bs,
                shuffle=True,
                num_workers=cfg.train.num_workers,
                persistent_workers=(cfg.train.num_workers > 0),
                pin_memory=(device != 'cpu')
            ),
            DataLoader(
                SortOfClevrDataset(str(val_path)),
                batch_size=cfg.train.val_bs,
                shuffle=False,
                num_workers=0,
                pin_memory=(device != 'cpu')
            ),
            DataLoader(
                SortOfClevrDataset(str(test_path)),
                batch_size=cfg.train.val_bs,
                shuffle=False,
                num_workers=0,
                pin_memory=(device != 'cpu')
            )
        )
    else:
        raise ValueError(f'unrecognised dataloader mode: {cfg.train.loader_mode}')