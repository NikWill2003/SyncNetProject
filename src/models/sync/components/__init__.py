"""The component library: the schema's six slots as swappable classes.

An ablation is a different class passed at construction, never a branch
taken at forward time (design principle P1). Two laws constrain every
composition -- L1: binding without competition collapses to one attractor;
L2: optimisation abandons phase wherever a content pathway can do its job.
"""

from .binding import CompetitiveClaim, GivenTokens, PartitionRead, QueryRead
from .dynamics import KuramotoStep, SkewGenerator, tangent
from .identity import Exchangeable, PhaseNative
from .interventions import anchor_shuffle, phase_shuffle
from .medium import PrivateLines, SharedBus, SilentBus
from .readout import HeadReadout, VoteReadout
