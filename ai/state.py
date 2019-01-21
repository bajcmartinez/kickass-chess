import chess
import numpy as np

class State:

    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def get_possible_moves(self):
        return list(self.board.legal_moves)

    def serialize_for_ai(self):
        # Make sure that the board is in a valid state, otherwise throw erorr
        assert self.board.is_valid()

        board_state = np.zeros(64, np.uint8)

        # First we save the board in a 8x8 grid (or vector in this case)
        for i in range(64):
            pp = self.board.piece_at(i)
            if pp is not None:
                board_state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
                                  "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}[pp.symbol()]

        # in order to store castling we are going to change the 'piece type' of the first or last column,
        # if I have the right to do castling queen side, the first column will switch from 4 (Rook) to 7 (Rook
        # with castling allowed) In the case of black pieces same thing, but with 15 instead of 7
        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert board_state[0] == 4
            board_state[0] = 7
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert board_state[7] == 4
            board_state[7] = 7
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert board_state[56] == 8+4
            board_state[56] = 15
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert board_state[63] == 8+4
            board_state[63] = 15

        # We reserve now number 8 for en passant
        if self.board.ep_square is not None:
            assert board_state[self.board.ep_square] == 0
            board_state[self.board.ep_square] = 8

        board_state = board_state.reshape(8, 8)
        state = np.zeros((5, 8, 8), np.uint8)

        # 0-3 columns to binary (eg 0010 represents 2)
        state[0] = (board_state >> 3) & 1
        state[1] = (board_state >> 2) & 1
        state[2] = (board_state >> 1) & 1
        state[3] = (board_state >> 0) & 1

        # Next we save who's next (1 means white, 0 means black)
        state[4] = self.board.turn * 1.0
        return state

if __name__ == "__main__":
  s = State()
  s.serialize_for_ai()