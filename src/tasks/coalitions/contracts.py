from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor


class CoalitionsBatch(TypedDict):
    streams: Tensor      
    commands: Tensor     
    targets: Tensor      
    loss_mask: Tensor    
    regime: Tensor       
    oracle_adj: Tensor   
    active_gid: Tensor  


class CoalitionsOutput(TypedDict):
    logits: Tensor                    
    traces: NotRequired[dict]        
                                    
