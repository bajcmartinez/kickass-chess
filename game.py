import time
import numpy as np
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, session, request
import chess
from ai.state import State
import traceback

graph = tf.get_default_graph()
model = load_model("ai/datasets/model.h5")


class ChessAI(object):
    def __call__(self, state, playing_as_white=True):
        global graph
        global model
        with graph.as_default():
            return self.value(state, playing_as_white)

    def value(self, state, playing_as_white):
        if state.board.is_game_over():
            result = state.board.result()
            if playing_as_white:
                if result == "1-0":
                    return float("inf")
                elif result == "0-1":
                    return float("-inf")
                else:
                    return 0
            else:
                if result == "1-0":
                    return float("-inf")
                elif result == "0-1":
                    return float("inf")
                else:
                    return 0

        global graph
        global model
        with graph.as_default():
            board = state.serialize_for_ai()
            output = model.predict(np.array([board, ]))
            if playing_as_white:
                return output[0][0]
            else:
                return output[0][1]


app = Flask(__name__)
app.config['SECRET_KEY'] = "kickass-chess-secret"


@app.route("/")
def home():
    """Home screen with the chess board and all functionality"""

    # Create a new state and save it in session
    session["fen"] = State().board.fen()

    # Render the page
    return render_template("home.html")


@app.route("/api/move", methods=["POST"])
def move():
    fen = session["fen"]
    state = State()
    state.board.set_fen(fen)
    source = request.form["source"]
    target = request.form["target"]

    move = chess.Move(chess.SQUARE_NAMES.index(source), chess.SQUARE_NAMES.index(target))
    if move is not None and move != "":
        assert move in state.board.legal_moves
        try:
            state.board.push(move)

            # now let the computer take it's turn
            if not state.board.is_game_over():
                ai_move(state)

            session["fen"] = state.board.fen()
            return app.response_class(
                response=state.board.fen(),
                status=200
            )
        except Exception:
            traceback.print_exc()
            return app.response_class(
                response="Error processing gameplay",
                status=500
            )


def minimax(state, ai, depth=1, maximizing_player=True, playing_as_white = True):
    if depth >= 5 or state.board.is_game_over():
        return ai(state, playing_as_white)

    if maximizing_player:
        value = float("-inf")
    else:
        value = float("inf")

    if depth == 1:
        moves = []

    # In order to make it faster, let's not review all the game options,
    # let's just review the top 10 for each step
    possible_moves = []

    # Once we are deep don't run all possibilities
    if depth >= 2:
        for move in state.get_possible_moves():
            state.board.push(move)
            possible_moves.append((ai(state, playing_as_white), move))
            state.board.pop()
            possible_moves = sorted(possible_moves, key=lambda x: x[0], reverse=maximizing_player)
        possible_moves = possible_moves[:10]

    else:
        for move in state.get_possible_moves():
            possible_moves.append((0, move))

    for move in [x[1] for x in possible_moves]:
        state.board.push(move)
        cval = minimax(state, ai, depth+1, not maximizing_player, playing_as_white=playing_as_white)
        state.board.pop()

        if depth == 1:
            moves.append((cval, move))

        if maximizing_player:
            value = max(value, cval)
        else:
            value = min(value, cval)

    if depth == 1:
        return value, moves
    else:
        return value


def eval_options(state):
    ai = ChessAI()

    start = time.time()
    val, moves = minimax(state, ai, playing_as_white=state.board.turn == chess.WHITE)
    best_moves = sorted(moves, key=lambda x: x[0], reverse=True)
    if len(best_moves) == 0:
        return None

    eta = time.time() - start
    print("AI move in %.3f second(s)" % (eta))

    print("top 3 options:")
    for i, m in enumerate(best_moves[0:3]):
        print("  ", m)

    return best_moves[0][1]


def ai_move(state):
    move = eval_options(state)
    state.board.push(move)


if __name__ == '__main__':
    ai = ChessAI()
    state = State()
    state.board.set_fen("4k3/2R5/2R5/8/8/8/8/3K4 w - - 0 1")

    move = eval_options(state)
