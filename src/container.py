from src.box import Box

ContainerID = int


class Container(Box):
    """Container class"""

    def __init__(
        self, number: int, weight_limit: int,
        length: int, width: int, height: int,
    ) -> None:
        """Create new container object

        Args:
            number (int): number of the container
            weight_limit (int): weight limit of the container
            length (int): length of the container
            width (int): width of the container
            height (int): height of the container
        """
        super().__init__(length, width, height)
        
        self.number = number
        self.weight_limit = weight_limit
