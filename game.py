import numpy as np
from keras.models import load_model
from flask import Flask, render_template, session, request
import chess
from ai.state import State
import traceback

graph = None

class ChessAI(object):
    def __init__(self):
        print('loading model...')
        self.model = load_model("ai/datasets/model.h5")
        import tensorflow as tf
        global graph
        graph = tf.get_default_graph()
        print('model loaded.')

    def __call__(self, state, white=True):
        global graph
        with graph.as_default():
            board = state.serialize_for_ai()
            output = self.model.predict(np.array([board, ]))

            if white:
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
            ai_move(state)

            session["fen"] = State().board.fen()
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

MAXVAL = 100000

def minimax(state, ai, depth=1, maximizing_player=True, playing_as_white = True):
    if depth >= 2 or state.board.is_game_over():
        return ai(state), None

    value = -MAXVAL
    best_move = None

    for move in state.get_possible_moves():
        state.board.push(move)
        cval, bm = minimax(state, ai, depth+1, not maximizing_player, playing_as_white=playing_as_white)
        state.board.pop()

        if cval > value:
            best_move = move
            value = cval

    return value, best_move

def eval_options(state):
    ai = ChessAI()
    val, best_move = minimax(state, ai, playing_as_white=state.board.turn == chess.WHITE)
    print('best move:', val, best_move)
    if best_move is None:
        best_move = state.get_possible_moves()[0]
    return best_move

def ai_move(state):
    move = eval_options(state)
    print('Move', move)
    state.board.push(move)

if __name__ == '__main__':
    ai = ChessAI()
    state = State()

    # for each possible move in the current state we evaluate our AI and we get the probabilities of winning, yeah!
    for move in state.get_possible_moves():
        state.board.push(move)
        print(move, ai(state))
        state.board.pop()