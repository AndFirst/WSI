from copy import deepcopy

from LAB_3.player import Player

possible_wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                 [0, 3, 6], [1, 4, 7], [2, 5, 8],
                 [0, 4, 8], [2, 4, 6]]


class Board:
    def __init__(self, players: tuple[Player, Player]) -> None:
        self.__init_board()
        self.__init_players(players)

    def wins(self, player: Player) -> bool:
        pattern = [player.mark] * 3
        for sample in possible_wins:
            result = []
            for i in sample:
                result.append(self.board[i])
            if result == pattern:
                return True
        return False

    def draw(self) -> bool:
        return len(self.get_possible_moves()) == 0 and not (self.wins(self.player_max) or self.wins(self.player_min))

    def is_terminal(self) -> bool:
        return self.wins(self.player_max) or self.wins(self.player_min) or self.draw()

    def heuristics(self) -> float:
        if self.wins(self.player_max):
            value = 5
        elif self.wins(self.player_min):
            value = -5
        else:
            value = 0
            # wages = [3, 2, 3, 2, 4, 2, 3, 2, 3]
            # for i, field in enumerate(self.board):
            #     if field == self.player_max.mark:
            #         value += wages[i]
            #     if field == self.player_min.mark:
            #         value -= wages[i]
        return value

    def get_possible_moves(self) -> list[int]:
        moves = []
        for i, field in enumerate(self.board):
            if field == " ":
                moves.append(i)
        return moves

    def __str__(self) -> str:
        return " {} | {} | {} \n" \
               "---+---+---\n" \
               " {} | {} | {} \n" \
               "---+---+---\n" \
               " {} | {} | {} \n".format(*self.board)

    def __init_board(self) -> None:
        self.board = [" " for _ in range(9)]

    def __init_players(self, players):
        self.player_max = players[0]
        self.player_min = players[1]


def generate_new_boards(board: Board, maximizing: bool) -> list[Board]:
    moves = board.get_possible_moves()
    player = "O" if maximizing else "X"
    new_boards = []
    for move in moves:
        new_board = deepcopy(board)
        new_board.board[move] = player
        new_boards.append(new_board)
    return new_boards
