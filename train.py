from match import Match
import sys
import chess.pgn
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras

promotions = {
    'q': 'queen',
    'r': 'rook',
    'b': 'bishop',
    'n': 'knight',
    '': ''
}

promotions_num = ['', 'q', 'r', 'b', 'n']

train_games = []
train_labels = []

def main(args):
    for arg in args:
        pgn = open(arg)
        game = chess.pgn.read_game(pgn)
        limit = 0
        while game is not None and limit < 10:
            if not game.errors:
                add_game(game)
            game = chess.pgn.read_game(pgn)
            limit += 1

    # crear el modelo
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # compilar el modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # entrenar el modelo
    model.fit(np.array(train_games), train_labels, epochs=5)

    # guardar el modelo
    model.save('deep_blue.h5')

def add_game(game):
    match = Match()
    board = game.board()
    for move in game.mainline_moves():
        the_board = match.board.numify()
        strmove = str(move)
        p = ''
        ox = 'abcdefgh'.find(strmove[0])
        oy = '87654321'.find(strmove[1])
        nx = 'abcdefgh'.find(strmove[2])
        ny = '87654321'.find(strmove[3])
        if len(str(move)) > 4:
            p = strmove[4]

        error = match.move(ox, oy, nx, ny, promotions[p])
        if error:
            return

        if match.board.white_turn:
            train_games.append(the_board)
            train_labels.append(move_to_num(ox, oy, nx, ny, p))


def move_to_num(ox, oy, nx, ny, p):
    return int(str(ox*10000 + oy*1000 + nx*100 + ny*10 + promotions_num.index(p)))

if __name__ == '__main__':
    main(sys.argv[1:])
