from math import inf

from LAB_3.board import Board, generate_new_boards


def min_max(board: Board, depth: int, maximizing: bool) -> float:
    if board.is_terminal() or depth == 0:
        return board.heuristics()

    if maximizing:
        best_value = -inf
        for new_board in generate_new_boards(board, maximizing):
            value = min_max(new_board, depth - 1, maximizing=False)
            if value > best_value:
                best_value = value
        return best_value
    else:
        best_value = +inf
        for new_board in generate_new_boards(board, maximizing):
            value = min_max(new_board, depth - 1, maximizing=True)
            if value < best_value:
                best_value = value
        return best_value
