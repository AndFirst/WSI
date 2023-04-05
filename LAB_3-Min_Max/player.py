class Player:
    def __init__(self, mark: str, maximize: bool, depth: int) -> None:
        self.mark = mark
        self.maximize = maximize
        self.depth = depth
