import json
import random
from collections import defaultdict
from LAB_3.board import Board, generate_new_boards
from LAB_3.min_max import min_max
from LAB_3.player import Player


def game(players_depths: tuple[int, int]):
    player_max = Player("O", True, players_depths[0])
    player_min = Player("X", False, players_depths[1])
    board = Board((player_max, player_min))
    player_max_moves = True

    while not board.is_terminal():
        results = defaultdict(list)
        if player_max_moves:
            depth = player_max.depth
            for move in generate_new_boards(board, maximizing=True):
                min_max_value = min_max(move, depth=depth, maximizing=False)
                results[min_max_value].append(move)
            new_board = random.choice(results[max(results)])
            board = new_board
        else:
            depth = player_min.depth
            for move in generate_new_boards(board, maximizing=False):
                min_max_value = min_max(move, depth=depth, maximizing=True)
                results[min_max_value].append(move)
            new_board = random.choice(results[min(results)])
            board = new_board
        print(board)
        player_max_moves = not player_max_moves
    if board.wins(board.player_max):
        print(f"{board.player_max.mark} win")
        winner = board.player_max.mark
    elif board.wins(board.player_min):
        print(f"{board.player_min.mark} win")
        winner = board.player_min.mark
    else:
        print("DRAW")
        winner = 'D'
    return winner


if __name__ == "__main__":
    all_data = {}
    players_intelligence = [(1, 1), (2, 1), (1, 2), (2, 2), (5, 1), (1, 5), (5, 5), (9, 1), (1, 9), (9, 9)]
    for intelligence in players_intelligence:
        game_result = {'O': 0,
                       'X': 0,
                       'D': 0}
        for i in range(1):
            win = game(intelligence)
            game_result[win] += 1
        all_data[str(intelligence)] = game_result
    with open("results.json", "w") as file:
        json.dump(all_data, file)
