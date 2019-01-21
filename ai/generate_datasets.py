import os
import chess.pgn
import numpy as np
from state import State

def generate_datasets(num_samples=None):
    statuses, winners = [], []
    values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    ng = 0
    # pgn files in the data folder
    for fn in os.listdir("data"):
        # open the PGN file, remember that each file may contain more than one game.
        pgn = open(os.path.join("data", fn))
        while 1:
            ng += 1
            print("Processing game:", ng)
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                continue
            # game was loaded
            if game is None:
                break

            # the game contains results
            res = game.headers['Result']
            if res not in values:
                continue

            value = values[res]
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                state = State(board).serialize_for_ai()
                statuses.append(state)
                winners.append(value)

            if num_samples is not None and len(statuses) > num_samples:
                return statuses, winners

    statuses = np.array(statuses)
    winners = np.array(winners)

    return statuses, winners

if __name__ == '__main__':
    statuses, winners = generate_datasets(1000000)
    np.savez("datasets/states.npz", statuses, winners)