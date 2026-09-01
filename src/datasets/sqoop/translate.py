from . import spec as C

def encode_question(x: str, rel: str, y: str) -> list[int]:
    return [C.SHAPE_TO_IDX[x], C.REL_TO_IDX[rel], C.SHAPE_TO_IDX[y]]

def decode_question(idxs) -> str:
    return ' '.join(C.IDX_TO_TOKEN[int(i)] for i in idxs)