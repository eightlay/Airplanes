from src.box import Box
from src.ems import EMS
from src.placement import Placement
from src.utils import construct_encoder_decoder

PACK_TYPES = ["без упаковки", "дер. ящик", '']
PACK_TYPES_ENCODER, PACK_TYPES_DECODER = construct_encoder_decoder(PACK_TYPES)

DANGEROUS_TYPES = ["неопасный", '']
DANGEROUS_TYPES_ENCODER, DANGEROUS_TYPES_DECODER = construct_encoder_decoder(
    DANGEROUS_TYPES
)


CargoID = int


class Cargo(Box):
    """Cargo class"""

    def __init__(
        self, name: str, weight: int,
        length: int, width: int, height: int,
        pack_type: str = '', dangerous_type: str = ''
    ) -> None:
        """Create new cargo object

        Args:
            name (str): name of the cargo
            weight (str): weight of the cargo
            length (int): length of the cargo
            width (int): width of the cargo
            height (int): height of the cargo
            pack_type (str): pack type of the cargo
            dangerous_type (str): dangerous type of the cargo
        """
        super().__init__(length, width, height)
        
        self.name = name
        self.weight = weight
        self.pack_type = PACK_TYPES_ENCODER[pack_type]
        self.dangerous = DANGEROUS_TYPES_ENCODER[dangerous_type]

    def valid_placements(self, ems: EMS) -> list[Placement]:
        # TODO
        pass