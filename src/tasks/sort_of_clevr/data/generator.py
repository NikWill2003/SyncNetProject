# TODO: need to add in reference to original github that I adapted

from __future__ import annotations

import cv2
import numpy as np
import random
from pathlib import Path
from typing import TYPE_CHECKING

from .constants import *

if TYPE_CHECKING:
    from ..config import SortOfClevrDataConfig

def center_generate(
        objects: list[np.ndarray], obj_size: int, img_size: int
        ) -> np.ndarray:
    
    while True:
        pas = True
        center = np.random.randint(0+obj_size, img_size - obj_size, 2)        
        if len(objects) > 0:
            for _, c, _ in objects:
                # squares are axis-aligned with half-width obj_size, so
                # they overlap unless they are separated along x OR y.
                # Euclidean distance alone allowed diagonal overlap
                # (~15% of scenes had an occluded object).
                if (abs(center[0] - c[0]) < obj_size * 2
                        and abs(center[1] - c[1]) < obj_size * 2):
                    pas = False
        if pas:
            return center
        

def generate_non_relational_question(
        objects: list[np.ndarray], img_size: int
        ) -> tuple[np.ndarray, int]:
    
    question = np.zeros((QUESTION_SIZE))
    color = random.randint(0, len(COLOURS) - 1)
    question[color] = 1
    question[Q_TYPE_IDX] = 1
    subtype = random.randint(0,2)
    question[subtype+SUB_Q_TYPE_IDX] = 1
    """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
    if subtype == 0:
        """query shape->rectangle/circle"""
        if objects[color][2] == 'r':
            answer = 2
        else:
            answer = 3

    elif subtype == 1:
        """query horizontal position->yes/no"""
        if objects[color][1][0] < img_size / 2:
            answer = 0
        else:
            answer = 1

    elif subtype == 2:
        """query vertical position->yes/no"""
        if objects[color][1][1] < img_size / 2:
            answer = 0
        else:
            answer = 1

    return question, answer


def generate_binary_question(objects: list[np.ndarray]) -> tuple[np.ndarray, int]:
    
    question = np.zeros((QUESTION_SIZE))
    color = random.randint(0,len(COLOURS)-1)
    question[color] = 1
    question[Q_TYPE_IDX+1] = 1
    subtype = random.randint(0,2)
    question[subtype+SUB_Q_TYPE_IDX] = 1

    if subtype == 0:
        """closest-to->rectangle/circle"""
        my_obj = objects[color][1]
        dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
        # exclude self by index: distances are SQUARED pixel distances
        # (up to ~11k), so the original 999 sentinel was reachable and
        # made an object its own nearest neighbour in ~35% of scenes.
        dist_list[color] = float('inf')
        closest = dist_list.index(min(dist_list))
        if objects[closest][2] == 'r':
            answer = 2
        else:
            answer = 3
            
    elif subtype == 1:
        """furthest-from->rectangle/circle"""
        my_obj = objects[color][1]
        dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
        furthest = dist_list.index(max(dist_list))
        if objects[furthest][2] == 'r':
            answer = 2
        else:
            answer = 3

    elif subtype == 2:
        """count->1~6"""
        my_obj = objects[color][2]
        count = -1
        for obj in objects:
            if obj[2] == my_obj:
                count +=1 
        answer = count+4

    return question, answer


def generate_ternary_question(
        objects: list[np.ndarray], t_subtype: int
        ) -> tuple[np.ndarray, int]:

    question = np.zeros((QUESTION_SIZE))
    rnd_colors = np.random.permutation(np.arange(len(COLOURS)))
    # 1st object
    color1 = rnd_colors[0]
    question[color1] = 1
    # 2nd object
    color2 = rnd_colors[1]
    question[len(COLOURS) + color2] = 1

    question[Q_TYPE_IDX + 2] = 1
    
    if t_subtype >= 0 and t_subtype < 3:
        subtype = t_subtype
    else:
        subtype = random.randint(0, 2)

    question[subtype+SUB_Q_TYPE_IDX] = 1

    # get coordiantes of object from question
    A = objects[color1][1]
    B = objects[color2][1]

    if subtype == 0:
        """between->1~4"""

        between_count = 0 
        # check is any objects lies inside the box
        for other_obj in objects:
            # skip object A and B
            if (other_obj[0] == color1) or (other_obj[0] == color2):
                continue

            # Get x and y coordinate of third object
            other_objx = other_obj[1][0]
            other_objy = other_obj[1][1]

            if (A[0] <= other_objx <= B[0] and A[1] <= other_objy <= B[1]) or \
                (A[0] <= other_objx <= B[0] and B[1] <= other_objy <= A[1]) or \
                (B[0] <= other_objx <= A[0] and B[1] <= other_objy <= A[1]) or \
                (B[0] <= other_objx <= A[0] and A[1] <= other_objy <= B[1]):
                between_count += 1

        answer = between_count + 4
    elif subtype == 1:
        """is-on-band->yes/no"""
        
        grace_threshold = 12  # half of the size of objects
        epsilon = 1e-10  
        m = (B[1]-A[1])/((B[0]-A[0]) + epsilon ) # add epsilon to prevent dividing by zero
        c = A[1] - (m*A[0])

        answer = 1  # default answer is 'no'

        # check if any object lies on/close the line between object A and object B
        for other_obj in objects:
            # skip object A and B
            if (other_obj[0] == color1) or (other_obj[0] == color2):
                continue

            other_obj_pos = other_obj[1]
            
            # y = mx + c
            y = (m*other_obj_pos[0]) + c
            if (y - grace_threshold)  <= other_obj_pos[1] <= (y + grace_threshold):
                answer = 0
    elif subtype == 2:
        """count-obtuse-triangles->1~6"""

        obtuse_count = 0

        for other_obj in objects:
            # skip object A and B
            if (other_obj[0] == color1) or (other_obj[0] == color2):
                continue

            # get position of 3rd object
            C = other_obj[1]
            # edge length
            a = np.linalg.norm(B - C)
            b = np.linalg.norm(C - A)
            c = np.linalg.norm(A - B)
            # angles by law of cosines; clip guards float error on
            # near-degenerate (collinear) triples, which otherwise gives
            # nan and is silently treated as not-obtuse
            alpha = np.rad2deg(np.arccos(
                np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c), -1.0, 1.0)))
            beta = np.rad2deg(np.arccos(
                np.clip((a ** 2 + c ** 2 - b ** 2) / (2 * a * c), -1.0, 1.0)))
            gamma = np.rad2deg(np.arccos(
                np.clip((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), -1.0, 1.0)))
            max_angle = max(alpha, beta, gamma)
            if max_angle >= 90 and max_angle < 180:
                obtuse_count += 1

        answer = obtuse_count + 4

    return question, answer


def generate_sample(
        img_size: int, obj_size: int, nb_questions: int, t_subtype: int
        ):
    
    objects = []
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # generate objects
    for color_id,color in enumerate(COLOURS.values()):  
        center = center_generate(objects, obj_size, img_size)
        if random.random()<0.5:
            start = (center[0]-obj_size, center[1]-obj_size)
            end = (center[0]+obj_size, center[1]+obj_size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id,center,'r'))
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, obj_size, color, -1)
            objects.append((color_id,center,'c'))

    ternary_questions = []
    binary_questions = []
    nonrel_questions = []

    ternary_answers = []
    binary_answers = []
    nonrel_answers = []
    
    # generate questions
    for _ in range(nb_questions):
        # NOTE (2026-08): these two calls had their second argument
        # swapped. generate_ternary_question takes t_subtype and was
        # receiving img_size (75 >= 3, so it silently fell through to the
        # random branch -- cfg.dataset.t_subtype had no effect);
        # generate_non_relational_question takes img_size and was
        # receiving t_subtype (-1, so the left-of-centre and top-half
        # tests compared against -0.5 and ALWAYS answered 'no').
        ternary_q, ternary_a = generate_ternary_question(objects, t_subtype)
        ternary_questions.append(ternary_q)
        ternary_answers.append(ternary_a)
        
        binary_q, binary_a = generate_binary_question(objects)
        binary_questions.append(binary_q)
        binary_answers.append(binary_a)

        nonrel_q, nonrel_a = generate_non_relational_question(objects, img_size)
        nonrel_questions.append(nonrel_q)
        nonrel_answers.append(nonrel_a)
    
    # keep uint8: 8x smaller on disk and in the GPU cache; the loader
    # divides by 255 (see _load_sort_of_clevr)
    return (
        img,
        (ternary_questions, ternary_answers),
        (binary_questions, binary_answers),
        (nonrel_questions, nonrel_answers),
        objects,
        )


def build_dataset(
        dataset_size: int, img_size: int, obj_size: int, 
        nb_questions: int, t_subtype: int
        ) -> dict[str, np.ndarray]:
    
    imgs = []
    ternary_questions, ternary_answers = [], []
    binary_questions, binary_answers = [], []
    nonrel_questions, nonrel_answers = [], []
    # ground-truth scene metadata: lets analysis split questions by
    # whether the queried objects share a quadrant (i.e. whether the
    # question actually requires cross-module communication) without
    # recovering object positions from pixels
    obj_positions, obj_shapes = [], []

    for _ in range(dataset_size):
        img, ternary, binary, nonrel, objects = generate_sample(
            img_size, obj_size, nb_questions, t_subtype
            )
        imgs.append(img)
        obj_positions.append([o[1] for o in objects])
        obj_shapes.append([1 if o[2] == 'r' else 0 for o in objects])
        ternary_questions.append(ternary[0])
        ternary_answers.append(ternary[1])
        binary_questions.append(binary[0])
        binary_answers.append(binary[1])
        nonrel_questions.append(nonrel[0])
        nonrel_answers.append(nonrel[1])

    
    return {
        'images': np.array(imgs),
        'ternary_questions': np.array(ternary_questions),
        'ternary_answers': np.array(ternary_answers),
        'binary_questions': np.array(binary_questions), 
        'binary_answers': np.array(binary_answers), 
        'nonrel_questions': np.array(nonrel_questions),
        'nonrel_answers': np.array(nonrel_answers),
        # (n_scenes, n_colours, 2) centres in (x, y); colour index == row
        'object_positions': np.array(obj_positions, dtype=np.int16),
        # (n_scenes, n_colours) 1 = rectangle, 0 = circle
        'object_shapes': np.array(obj_shapes, dtype=np.uint8),
    }

def save_dataset(dataset: dict[str, np.ndarray], data_dir: Path, name: str):

    data_dir.mkdir(exist_ok=True, parents=True)
    file = data_dir / name
    # allow_pickle is a keyword-only arg of savez_compressed on numpy>=2.
    # All arrays here are numeric, so False enforces that: if a ragged or
    # object-dtype array is ever added it fails here, at write time,
    # instead of silently pickling and failing later at load.
    np.savez_compressed(file, allow_pickle=False, **dataset)

    print(f'saved {name} to {str(str(data_dir.absolute()))}')

def prepare_sort_of_clevr(cfg: SortOfClevrDataConfig) -> None:
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_dir = Path(cfg.root) / cfg.dir
    data_dir.mkdir(exist_ok=True, parents=True)

    print('building test datasets...')
    test_dataset = build_dataset(
        cfg.test_size, cfg.img_size, cfg.obj_size, 
        cfg.nb_questions, cfg.t_subtype
    )
    save_dataset(test_dataset, data_dir, f'test.npz')

    print('building validation datasets...')
    val_dataset = build_dataset(
        cfg.test_size, cfg.img_size, cfg.obj_size, 
        cfg.nb_questions, cfg.t_subtype
    )
    save_dataset(val_dataset, data_dir, f'val.npz')

    print('building train datasets...')
    train_dataset = build_dataset(
        cfg.train_size, cfg.img_size, cfg.obj_size, 
        cfg.nb_questions, cfg.t_subtype
    )
    save_dataset(train_dataset, data_dir, f'train.npz')