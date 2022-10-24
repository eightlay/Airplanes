class Box:
    """Box class"""

    def __init__(self, length: int, width: int, height: int) -> None:
        """Create new box object

        Args:
            length (int): length of the box
            width (int): width of the box
            height (int): height of the box
        """
        self.length = length
        self.width = width
        self.height = height
        self.volume = length * width * height

    @property
    def dims(self) -> tuple[int, int, int, int]:
        """Get length, width, height and volume of the box

        Returns:
            tuple[int, int, int, int]: _description_
        """
        return self.length, self.width, self.height, self.volume
