# Edge relaxation family: the paper's EBC algorithm plus all scoring variants.
from .ebc import EdgeRelaxation
from .crossing import EdgeRelaxationCrossing
from .stress import EdgeRelaxationStress
from .angular import EdgeRelaxationAngular
from .currentflow import EdgeRelaxationCurrentFlow
from .embeddedness import EdgeRelaxationEmbeddedness
from .community import EdgeRelaxationCommunity
from .fiedler import EdgeRelaxationFiedler
from .adaptive import EdgeRelaxationAdaptive
