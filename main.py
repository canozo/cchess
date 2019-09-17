from match import Match
from pyglet.window import mouse
from pyglet.window import key
import pyglet
import itertools

game_window = pyglet.window.Window(height=512, width=512)
message_label = pyglet.text.Label(font_name='Times New Roman',
                                  font_size=36,
                                  color=(35, 203, 35, 255),
                                  x=game_window.width // 2,
                                  y=game_window.height // 2,
                                  anchor_x='center',
                                  anchor_y='center')
board_normal = pyglet.sprite.Sprite(pyglet.image.load('resources/board-normal.png'))
board_imgs = [[None for _ in range(8)] for _ in range(8)]
piece_held = None
match = Match()
ai_mode = True
promotion = None
old_pos = (0, 0)


def update_board():
    for i, j in itertools.product(range(8), repeat=2):
        piece = match.board.chessboard[i][j]
        if piece == '':
            continue
        if piece.isupper():
            piece_name = f'{piece}w'
        else:
            piece_name = f'{piece}b'
        board_imgs[i][j] = pyglet.sprite.Sprite(pyglet.image.load(f'resources/{piece_name}.png'))


@game_window.event
def on_draw():
    game_window.clear()
    board_normal.draw()
    chessboard = match.board.chessboard
    for x, y in itertools.product(range(8), repeat=2):
        if chessboard[y][x] != '':
            piece = board_imgs[y][x]
            if piece != piece_held:
                piece.x = x * 64
                piece.y = 448 - y * 64
            piece.draw()

    gameover, stalemate = match.status()
    if stalemate:
        message_label.text = 'Stalemate!'
        message_label.draw()
    elif gameover:
        if match.board.white_turn:
            message_label.text = 'Checkmate. Black won!'
        else:
            message_label.text = 'Checkmate. White won!'
        message_label.draw()


@game_window.event
def on_mouse_press(x, y, button, modifiers):
    global piece_held, old_pos
    if button == mouse.LEFT and not match.gameover:
        chessboard = match.board.chessboard
        piece = chessboard[7 - y//64][x//64]
        if piece != '':
            piece_held = board_imgs[7 - y//64][x//64]
            old_pos = (x//64, 7 - y//64)


@game_window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if piece_held is not None:
        piece_held.x = x - 32
        piece_held.y = y - 32


@game_window.event
def on_mouse_release(x, y, button, modifiers):
    global piece_held
    if piece_held is not None:
        dx, dy = x//64, 7 - y//64
        ox, oy = old_pos
        error = match.move(ox, oy, dx, dy, promotion)
        update_board()
        if not error and ai_mode:
            match.ai_move()
            update_board()

    piece_held = None


@game_window.event
def on_text(text):
    if text == 'm':
        match.ai_move()
    elif text == 'u':
        match.undo()
        update_board()


@game_window.event
def on_key_press(symbol, modifiers):
    global match, promotion, ai_mode
    if symbol == key.N:
        match = Match()
        update_board()
    elif symbol == key.A:
        ai_mode = not ai_mode
    elif symbol == key.Q:
        promotion = 'queen'
    elif symbol == key.B:
        promotion = 'bishop'
    elif symbol == key.R:
        promotion = 'rook'
    elif symbol == key.K:
        promotion = 'knight'


# @game_window.event
# def on_key_release(symbol, modifiers):
#     global promotion
#     if symbol in (key.Q, key.B, key.R, key.K):
#         promotion = None


def main():
    update_board()
    pyglet.app.run()


if __name__ == '__main__':
    main()
