from match import Match
import sys
import chess.pgn
import io
import numpy as np
import tensorflow as tf
from utils import *
from ops import *

tests = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]])

tests_labels = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

promotions = {
    '': '',
    'q': 'queen',
    'r': 'rook',
    'b': 'bishop',
    'n': 'knight'
}

promotions_num = ['', 'q', 'r', 'b', 'n']

move_dict = {
    '46440': 0,
    '41430': 1,
    '67550': 2,
    '10220': 3,
    '57240': 4,
    '50230': 5
}

train_games = []
train_labels = []

def train(args):
    for arg in args:
        pgn = open(arg)
        game = chess.pgn.read_game(pgn)
        limit = 0
        while game is not None and limit < 1000:
            if not game.errors:
                add_game(game)
            game = chess.pgn.read_game(pgn)
            limit += 1

    # tf
    data = np.array(train_games)
    labels = np.array(train_labels)
    cosacosa(data[:900], labels[:900], data[100:], labels[100:], tests, tests_labels)

def add_game(game):
    count = 0
    match = Match()
    board = game.board()
    for move in game.mainline_moves():
        if count >= 6:
            # modo_puercada = on
            return

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

        count += 1
        tp_move = f'{ox}{oy}{nx}{ny}{promotions_num.index(p)}'
        temp = [0, 0, 0, 0, 0, 0]
        temp[move_dict[tp_move]] = 1
        train_games.append(the_board)
        train_labels.append(temp)

def cosacosa(x_train, y_train, x_valid, y_valid, x_test, y_test):
    img_h = img_w = 8
    img_size_flat = img_h * img_w
    n_classes = 6

    print(f'x_train.shape: {x_train.shape}')
    print(f'y_train.shape: {y_train.shape}')

    print(f'x_test.shape: {x_test.shape}')
    print(f'y_test.shape: {y_test.shape}')

    # Hyper-parameters
    learning_rate = 0.001  # The optimization initial learning rate
    epochs = 20  # Total number of training epochs
    batch_size = 100  # Training batch size
    display_freq = 100  # Frequency of displaying the training results

    # Create the graph for the linear model
    # Placeholders for inputs (x) and outputs(y)
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

    # create weight matrix initialized randomely from N~(0, 0.01)
    W = weight_variable(shape=[img_size_flat, n_classes])

    # create bias vector initialized as zero
    b = bias_variable(shape=[n_classes])

    output_logits = tf.matmul(x, W)
    y_pred = tf.nn.softmax(output_logits)

    # Model predictions
    cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

    # Define the loss function, optimizer, and accuracy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # Creating the op for initializing all variables
    init = tf.global_variables_initializer()

    # Launch the graph (session)
    with tf.Session() as sess:
        sess.run(init)
        global_step = 0
        # Number of training iterations in each epoch
        num_tr_iter = int(len(y_train) / batch_size)
        for epoch in range(epochs):
            print('Training epoch: {}'.format(epoch + 1))
            x_train, y_train = randomize(x_train, y_train)
            for iteration in range(num_tr_iter):
                global_step += 1
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

                # Run optimization op (backprop)
                feed_dict_batch = {x: x_batch, y: y_batch}
                sess.run(optimizer, feed_dict=feed_dict_batch)

                if iteration % display_freq == 0:
                    # Calculate and display the batch loss and accuracy
                    loss_batch, acc_batch = sess.run([loss, accuracy],
                                                    feed_dict=feed_dict_batch)

                    print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                        format(iteration, loss_batch, acc_batch))

            # Run validation after every epoch
            feed_dict_valid = {x: x_valid[:1000], y: y_valid[:1000]}
            loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
            print('---------------------------------------------------------')
            print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
                format(epoch + 1, loss_valid, acc_valid))
            print('---------------------------------------------------------')

        # Test the network after training
        # Accuracy
        feed_dict_test = {x: tests, y: tests_labels}
        loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
        print('---------------------------------------------------------')
        print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
        print('---------------------------------------------------------')

        pred = sess.run(cls_prediction, feed_dict={x: tests})
        print(pred)
        # pred = sess.run(correct_prediction, feed_dict={x: tests})

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('debe enviar nombre del archivo PGN como arg')
    else:
        train(sys.argv[1:])
