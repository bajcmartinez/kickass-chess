import numpy as np
from keras.models import load_model
from state import State

class ChessAI(object):
    def __init__(self):
        self.model = load_model("datasets/model.h5")

    def __call__(self, state, white=True):
        board = state.serialize_for_ai()
        output = self.model.predict(np.array([board, ]))
        if white:
            return output[0][0]
        else:
            return output[0][1]

if __name__ == '__main__':
    ai = ChessAI()
    state = State()

    # for each possible move in the current state we evaluate our AI and we get the probabilities of winning, yeah!
    for move in state.get_possible_moves():
        state.board.push(move)
        print(move, ai(state))
        state.board.pop()