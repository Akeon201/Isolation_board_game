import copy
import bisect
import random
import tkinter as tk
import time

# Author: Kenyon Leblanc

#######################################################################################################################
# SETTINGS

# True: Both players are AI
# False: User plays player 1 (Blue tile)
both_ai = True

# Depth level for minimax algorithm
depth = 3

# Who starts the game first
starting_player = random.choice(['P1', 'P2'])  # random.choice(['P1', 'P2'])

# 1 for basic heuristic: random best tile around player
# 2 for blocking heuristic: random blocking tile, if none is found will fall back to random best tile around player
p1_heuristic = 1
p2_heuristic = 2

# Print how long it took AI to take decide a move
print_time = False

# Print moves made
print_moves = False

#######################################################################################################################


class IsolationGame:
    """
        Class for game logic
    """

    def __init__(self):
        # Initialize game board as 8x6
        self.board = [['\n'] * 6 for _ in range(8)]
        self.board[0][3] = 'P2'
        self.board[7][2] = 'P1'

        # Player positions
        self.p2_pos = (0, 3)
        self.p1_pos = (7, 2)

        # Starting player
        self.current_player = starting_player

    def ai_move(self):
        """
        Finds a move and moves AI
        :return: Game state used in moving player. Use later for removing token
        """
        start_time = time.time()
        score, move = self.minimax(self.get_state(), depth, True, float('-inf'), float('inf'))
        end_time = time.time()

        duration = end_time - start_time
        if print_time:
            print(f"{self.current_player} took {duration:.5f} seconds to move.")

        if self.current_player == 'P2':
            x, y = move['p2_pos']
            self.move('P2', x, y)
        else:
            x, y = move['p1_pos']
            self.move('P1', x, y)

        return move

    def print_board(self, board):
        """
        Print board state nicely in console
        :param board: Board state
        """
        for row in board:
            print(row)

    def get_state(self):
        """
        Takes current board state and returns it
        :return: game state
        """
        new_board = copy.deepcopy(self.board)

        return {'board': new_board, 'p1_pos': self.p1_pos, 'p2_pos': self.p2_pos, 'p2_old_pos': self.p2_pos,
                'p1_old_pos': self.p1_pos, 'p1_removed_token': (0, 0), 'p2_removed_token': (0, 0), 'total_heuristic': 0}

    def shuffle_max_heuristic_states(self, states):
        """
        Randomly decide ties if there are more than one state with same highest heuristic value. Only shuffles highest same scoring states.
        States must be sorted in descending order.
        :param states: All game states
        :return: shuffled states
        """
        max_heuristic = states[0]['total_heuristic']

        # Count states with the maximum heuristic
        count_max = sum(1 for state in states if state['total_heuristic'] == max_heuristic)

        # Shuffle top states
        top_states = states[:count_max]
        random.shuffle(top_states)

        return top_states + states[count_max:]

    def minimax(self, state, depth, maximizing_player, alpha, beta):
        """
        minimax algorithm for searching for best move
        :param state: game state
        :param depth: how far to search
        :param maximizing_player: which player is max
        :param alpha: negative inf
        :param beta: positive inf
        :return: heuristic value and best move state
        """
        if depth == 0 or self.is_terminal_state(state) == 0:
            return state['total_heuristic'], state

        if self.current_player == 'P1':
            other_player = 'P2'
        else:
            other_player = 'P1'

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for child in self.get_all_surrounding_player_tile_states(state, self.current_player):
                eval, _ = self.minimax(child, depth - 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = child
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move

        else:
            min_eval = float('inf')
            best_move = None
            for child in self.get_all_surrounding_player_tile_states(state, other_player):
                eval, _ = self.minimax(child, depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = child
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def is_terminal_state(self, state):
        """
        check if game is over
        :param state: game state
        :return: 0 if game over, 1 if not
        """
        return self.is_game_over(state)

    def get_all_possible_states(self, state, player):
        """
        get every single possible board state that exists
        :param state: game state
        :param player: player simulating their turn
        :return: all states
        """
        possible_states = []

        # All player moves
        for move in self.get_moves(player, state):
            x, y = move

            # Removing tokens
            for i in range(8):
                for j in range(6):
                    if (state['board'][i][j] == '\n' and (i, j) != move) or state['board'][i][j] == player:
                        new_state = copy.deepcopy(state)
                        new_state = self.move(player, x, y, new_state)
                        new_state = self.remove_token(player, i, j, new_state)
                        new_state['total_heuristic'] = new_state['total_heuristic'] + self.evaluate_state(new_state)
                        # Sort states by heuristic value in descending value
                        bisect.insort(possible_states, new_state, key=lambda x: -x['total_heuristic'])

        # Shuffle top same heuristic values. To break ties.
        possible_states = self.shuffle_max_heuristic_states(possible_states)

        return possible_states

    def get_all_surrounding_player_tile_states(self, state, player):
        """
        Same logic as get_all_possible_states() except it limits the states to tiles around player
        :param state: game state
        :param player: player simulating their turn
        :return: all states
        """
        possible_states = []

        # player moves
        for move in self.get_moves(player, state):
            x, y = move

            new_state = self.move(player, x, y, state)
            if player == 'P1':
                opp_surr_tiles = self.get_moves('P2', new_state)
            else:
                opp_surr_tiles = self.get_moves('P1', new_state)
            new_state = self.revert_move(player, state)

            # removing tokens
            for tile in opp_surr_tiles:
                tile_x, tile_y = tile
                if (state['board'][tile_x][tile_y] == '\n' and (tile_x, tile_y) != move):
                    new_state = copy.deepcopy(state)
                    new_state = self.move(player, x, y, new_state)
                    new_state = self.remove_token(player, tile_x, tile_y, new_state)
                    new_state['total_heuristic'] = new_state['total_heuristic'] + self.evaluate_state(new_state)
                    # Sort states by heuristic value in descending value
                    bisect.insort(possible_states, new_state, key=lambda x: -x['total_heuristic'])

        # Shuffle top same heuristic values. To break ties.
        try:
            possible_states = self.shuffle_max_heuristic_states(possible_states)
        except IndexError:
            pass

        return possible_states

    def evaluate_state(self, state):
        """
        look at state and give it a heuristic value
        :param state: game state
        :return: total combined values of movement heuristic and token heuristic
        """
        if self.current_player == "P1":
            if p1_heuristic == 1:
                # Heuristic 1
                heuristic_1 = self.heuristic_move(state['p1_old_pos'], state['p1_pos'], state)
                heuristic_2 = self.heuristic_token(state['p2_pos'], state['p2_pos'], state)
            else:
                # Heuristic 2
                heuristic_1 = self.heuristic_move(state['p1_old_pos'], state['p1_pos'], state)
                heuristic_2 = self.heuristic_token_second_version(state['p2_pos'], state['p2_pos'], state)
        else:
            if p2_heuristic == 1:
                # Heuristic 1
                heuristic_1 = self.heuristic_move(state['p2_old_pos'], state['p2_pos'], state)
                heuristic_2 = self.heuristic_token(state['p1_pos'], state['p1_pos'], state)
            else:
                # Heuristic 2
                heuristic_1 = self.heuristic_move(state['p2_old_pos'], state['p2_pos'], state)
                heuristic_2 = self.heuristic_token_second_version(state['p1_pos'], state['p1_pos'], state)

        return heuristic_1 + heuristic_2

    def player_1_position(self):
        """
        returns P1 position
        :return: position
        """
        return self.p1_pos

    def player_2_position(self):
        """
        returns P2 position
        :return: position
        """
        return self.p2_pos

    def revert_move(self, player, state):
        """
        revert last move for advanced analysis in evaluation function
        :param player: which player to revert
        :param state: game state
        :return: modified game state
        """
        new_state = copy.deepcopy(state)

        if player == 'P1':
            old_x, old_y = new_state['p1_old_pos']
            new_x, new_y = new_state['p1_pos']
            new_state['board'][old_x][old_y] = 'P1'
            new_state['board'][new_x][new_y] = '\n'
        else:
            old_x, old_y = new_state['p2_old_pos']
            new_x, new_y = new_state['p2_pos']
            new_state['board'][old_x][old_y] = 'P2'
            new_state['board'][new_x][new_y] = '\n'

        return new_state

    def revert_token(self, player, state):
        """
        revert last token removal for advanced analysis in evaluation function
        :param player: which player token removal to revert
        :param state: game state
        :return: modified game state
        """
        new_state = copy.deepcopy(state)

        if player == 'P1':
            x, y = new_state['p1_removed_token']
            new_state['board'][x][y] = '\n'
        else:
            x, y = new_state['p2_removed_token']
            new_state['board'][x][y] = '\n'

        return new_state

    def heuristic_move(self, current, next, state):
        """
        Give a value to game state based on available tiles to move to
        :param current: (x, y) of before the move
        :param next: (x, y) of next move
        :param state: game state
        :return: heuristic value
        """
        x, y = next
        next_value = self.count_surrounding_cells(x, y, state)

        if self.current_player == 'P1':
            reverted_state = self.revert_move('P1', state)
            reverted_state = self.revert_token('P1', reverted_state)
        else:
            reverted_state = self.revert_move('P2', state)
            reverted_state = self.revert_token('P2', reverted_state)

        x, y = current
        current_value = self.count_surrounding_cells(x, y, reverted_state)

        # current value is next move because of the way implemented in get_states function. next_value is the previous move.
        if next_value > current_value:
            return 1
        elif next_value == current_value:
            return 0
        else:
            return -1

    def heuristic_token(self, opp_before_removed_token, opp_after_removed_token, state):
        """
        Give a value to a game state based on tiles around opponent when removing tile
        :param opp_before_removed_token: Current (x, y) of opponent
        :param opp_after_removed_token: Current (x, y) of opponent
        :param state: game state
        :return: heuristic value; 100 if no moves, 2 if less movement places after tile removal
        """
        x, y = opp_after_removed_token
        after_count = self.count_surrounding_cells(x, y, state)

        if self.current_player == 'P1':
            reverted_state = self.revert_token('P1', state)
        else:
            reverted_state = self.revert_token('P2', state)

        x, y = opp_before_removed_token
        before_count = self.count_surrounding_cells(x, y, reverted_state)

        if after_count == 0:
            return 100
        elif after_count < before_count:
            return 2
        else:
            return 0

    def heuristic_token_second_version(self, opp_before_removed_token, opp_after_removed_token, state):
        """
        Give a value to a game state based on tiles around opponent when removing tile. This one gives more values to tiles that lead to seclusion.
        :param opp_before_removed_token: Current (x, y) of opponent
        :param opp_after_removed_token: Current (x, y) of opponent
        :param state: game state
        :return: heuristic value; 100 if no moves, 4 if tile blocks movement path, 2 if less movement places after tile removal
        """
        x, y = opp_after_removed_token
        after_count = self.count_surrounding_cells(x, y, state)

        if self.current_player == 'P1':
            removed_tile = state['p1_removed_token']
            reverted_state = self.revert_token('P1', state)
        else:
            removed_tile = state['p2_removed_token']
            reverted_state = self.revert_token('P2', state)

        x, y = opp_before_removed_token
        before_count = self.count_surrounding_cells(x, y, reverted_state)

        if after_count == 0:
            return 100
        elif (after_count < before_count) and self.is_blocking_tile(removed_tile, state):
            return 4
        elif after_count < before_count:
            return 2
        else:
            return 0

    def is_blocking_tile(self, tile, state):
        """
        Looks at a 2 x 2 quadrant and returns 1 if it blocks opponent path
        :param tile: (x, y) of removed tile
        :param state: game state
        :return: 1 if found, 0 if nothing found
        """
        x, y = tile
        perimeter_tiles = self.get_perimeter_tiles()
        if self.current_player == 'P1':
            other_player = 'P2'
            other_player_x, other_player_y = state['p2_pos']
        else:
            other_player = 'P1'
            other_player_x, other_player_y = state['p1_pos']

        # Block opponent when not on perimeter tile
        try:
            if state['board'][x + 1][y] == '' and state['board'][x][y + 1] == '' and state['board'][x + 1][y + 1] != '':
                return 1
        except IndexError:
            pass

        try:
            if state['board'][x + 1][y] == '' and state['board'][x][y - 1] == '' and state['board'][x + 1][y - 1] != '':
                return 1
        except IndexError:
            pass

        try:
            if state['board'][x - 1][y] == '' and state['board'][x][y - 1] == '' and state['board'][x - 1][y - 1] != '':
                return 1
        except IndexError:
            pass

        try:
            if state['board'][x - 1][y] == '' and state['board'][x][y + 1] == '' and state['board'][x - 1][y + 1] != '':
                return 1
        except IndexError:
            pass

        # Block path on perimeter tile
        if (x, y) in perimeter_tiles:
            if y < 3 and state['board'][x][y + 1] == '' and state['board'][other_player_x][other_player_y + 1] == '':
                return 1
            elif y > 3 and state['board'][x][y - 1] == '' and state['board'][other_player_x][other_player_y - 1] == '':
                return 1
            elif x < 3 and state['board'][x + 1][y] == '' and state['board'][other_player_x + 1][other_player_y] == '':
                return 1
            elif x > 3 and state['board'][x - 1][y] == '' and state['board'][other_player_x - 1][other_player_y] == '':
                return 1

        return 0

    def get_perimeter_tiles(self):
        """
        list of perimeter tiles
        :return: list of perimeter tiles
        """
        perimeter_tiles = []

        # Top and bottom rows
        for i in range(6):
            perimeter_tiles.append((0, i))
            perimeter_tiles.append((7, i))

        # Left and right columns
        for j in range(1, 7):
            perimeter_tiles.append((j, 0))
            perimeter_tiles.append((j, 5))

        return perimeter_tiles

    def count_surrounding_cells(self, x, y, state):
        """
        Counts number of surrounding tiles with tokens
        :param x: row value
        :param y: column value
        :param state: game state
        :return: number of tiles w/ tokens
        """
        possible_moves = [
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
            (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)
        ]

        valid_moves = [
            (px, py) for px, py in possible_moves
            if (0 <= px < 8 and 0 <= py < 6 and state['board'][px][py] == '\n')
        ]

        return len(valid_moves)

    def get_moves(self, player, state=None):
        """
        get valid moves a player can make
        :param player: player you want to get moves for
        :param state: board state
        :return: all (x, y) moves
        """
        if state is None:
            x, y = self.p1_pos if player == 'P1' else self.p2_pos

            possible_moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                              (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]

            valid_moves = []
            for px, py in possible_moves:
                # Adjusted: New board boundaries
                if 0 <= px < 8 and 0 <= py < 6 and self.board[px][py] == '\n':
                    valid_moves.append((px, py))
            return valid_moves
        else:
            x, y = state['p1_pos'] if player == 'P1' else state['p2_pos']

            possible_moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                              (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)]

            valid_moves = []
            for px, py in possible_moves:
                if 0 <= px < 8 and 0 <= py < 6 and state['board'][px][py] == '\n':
                    valid_moves.append((px, py))
            return valid_moves

    def move(self, player, new_x, new_y, state=None):
        """
        move player to tile x, y
        :param player: player to move
        :param new_x: row
        :param new_y: column
        :param state: game state
        :return: game state modified
        """
        if state is None:
            # Validate the move
            if (new_x, new_y) not in self.get_moves(player):
                # print(f'Invalid move attempted: {new_x}, {new_y} by {player}')  # Add this line
                raise ValueError("Invalid move")

            # Update the player's position
            x, y = self.p1_pos if player == 'P1' else self.p2_pos
            if self.p2_pos == (0, 3) and self.current_player == 'P2':
                self.board[x][y] = ''
            elif self.p1_pos == (7, 2) and self.current_player == 'P1':
                self.board[x][y] = ''
            else:
                self.board[x][y] = '\n'

            self.board[new_x][new_y] = player
            if player == 'P1':
                self.p1_pos = (new_x, new_y)
            else:
                self.p2_pos = (new_x, new_y)
        else:
            if player == 'P2' and state['p2_pos'] == (0, 3):
                state['board'][0][3] = ''
                state['board'][new_x][new_y] = 'P2'
                state['p2_pos'] = (new_x, new_y)
            elif player == 'P1' and state['p1_pos'] == (7, 2):
                state['board'][7][2] = ''
                state['board'][new_x][new_y] = 'P1'
                state['p1_pos'] = (new_x, new_y)
            else:
                if player == 'P1':
                    px, py = state['p1_pos']
                    state['board'][px][py] = '\n'
                    state['p1_old_pos'] = state['p1_pos']
                    state['p1_pos'] = (new_x, new_y)
                    state['board'][new_x][new_y] = 'P1'
                else:
                    px, py = state['p2_pos']
                    state['board'][px][py] = '\n'
                    state['p2_old_pos'] = state['p2_pos']
                    state['p2_pos'] = (new_x, new_y)
                    state['board'][new_x][new_y] = 'P2'
            return state

    def remove_token(self, player, token_x, token_y, state=None):
        """
        remove a token from x, y
        :param player: player removing token
        :param token_x: row
        :param token_y: column
        :param state: game state
        :return: game state if nont None
        """
        if state is None:
            # Remove a token
            if self.board[token_x][token_y] != '\n':
                print(
                    f"Attempt to remove token by {player} at ({token_x}, {token_y}) with value: {self.board[token_x][token_y]}")
                raise ValueError("Invalid token removal")
            self.board[token_x][token_y] = ''
            self.current_player = 'P2' if player == 'P1' else 'P1'
        else:
            # Remove a token
            # print(f"Attempt to remove token by {player} at ({token_x}, {token_y}) with value: {self.board[token_x][token_y]}")
            state['board'][token_x][token_y] = ''
            if player == 'P1':
                state['p1_removed_token'] = (token_x, token_y)
            else:
                state['p2_removed_token'] = (token_x, token_y)
            return state

    def is_game_over(self, state=None):
        """
        check if game is over
        :param state: game state
        :return: 0 if game over, 1 if not
        """
        if state is None:
            return len(self.get_moves('P1')) == 0 or len(self.get_moves('P2')) == 0
        else:
            if len(self.get_moves('P1', state)) == 0 or len(self.get_moves('P2', state)) == 0:
                return 0
            else:
                return 1

    def get_winner(self):
        """
        check if winner is found
        :return: 0 if found, None if not
        """
        if self.is_game_over():
            return 'P2' if len(self.get_moves('P1')) == 0 and self.current_player == 'P1' else 'P1'
        return None


class GameOverDialog(tk.Toplevel):
    """
    when game ends, display this gui over game window
    """

    def __init__(self, parent, winner, restart_callback, exit_callback):
        super().__init__(parent)
        self.overrideredirect(True)

        self.geometry("400x200")
        self.configure(bg="white")  # white background color

        # larger font size and background color for the label
        label = tk.Label(self, text=f"Player {winner[-1]} wins!", font=("Arial", 16), bg="white")
        label.pack(pady=20, padx=20, fill="both", expand=True)

        # background color for the buttons
        restart_button = tk.Button(self, text="Restart", command=restart_callback, bg="white", font=("Arial", 14))
        restart_button.pack(pady=10, padx=15, fill="x")

        exit_button = tk.Button(self, text="Exit", command=exit_callback, bg="white", font=("Arial", 14))
        exit_button.pack(pady=10, padx=15, fill="x")

        self.center_window()

        self.transient(parent)
        self.grab_set()

    def center_window(self):
        """
        center gui window
        """
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))


class IsolationApp(tk.Tk):
    """
    gui to display game board
    """

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.create_widgets()
        self.update_board()
        self.move_phase = True
        self.moves = 0

    def center_window(self):
        """
        center game window
        """
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def create_widgets(self):
        """
        create widgets for window
        """
        tile_height = 4
        tile_width = 10

        self.title("Isolation")

        self.buttons = [
            [tk.Button(self, height=tile_height, width=tile_width, command=lambda i=i, j=j: self.on_click(i, j))
             for j in range(6)] for i in range(8)]

        for i, row in enumerate(self.buttons):
            for j, button in enumerate(row):
                button.grid(row=i, column=j)

        if self.game.current_player == "P1":
            self.label = tk.Label(self, text="Player 1's Move")
        else:
            self.label = tk.Label(self, text="Player 2's Move")

        self.label.grid(row=8, column=0, columnspan=6)

    def announce_winner(self, winner):
        """
        display window with winner, restart button, and exit button
        :winner: who has no moves left, if tie then who made last move to seclude opponent
        """

        def restart():
            dialog.destroy()
            self.destroy()
            start_game()

        def exit_app():
            self.destroy()  # Exit the application
            exit(0)

        if print_moves:
            print(self.moves)

        dialog = GameOverDialog(self, winner, restart, exit_app)
        dialog.mainloop()

    def on_click(self, i=None, j=None):
        """
        logic when board tiles are pressed
        :param i: row
        :param j: column
        """
        if self.game.is_game_over():
            winner = self.game.get_winner()
            self.announce_winner(winner)
            return

        player = self.game.current_player

        self.update_board()

        if self.game.current_player == 'P1':
            if both_ai != True:
                # Human player logic
                if self.move_phase:
                    moves = self.game.get_moves(player)
                    if (i, j) in moves:
                        self.move_phase = False  # Switch to token removal phase
                        self.game.move(player, i, j)
                        self.update_board()

                        if self.game.is_game_over():
                            winner = self.game.get_winner()
                            self.announce_winner(winner)
                            return

                        # change bottom gui text
                        if self.game.current_player == "P1":
                            self.label["text"] = f"Player 1 Remove a Tile"
                        else:
                            self.label["text"] = f"Player 2 Remove a Tile"
                else:
                    # Token removal
                    if self.game.board[i][j] == "\n":
                        self.game.remove_token(player, i, j)
                        self.move_phase = True  # Switch to movement phase

                        self.update_board()

                        if self.game.is_game_over():
                            winner = self.game.get_winner()
                            self.announce_winner(winner)
                        else:
                            if self.game.current_player == "P1":
                                self.label["text"] = f"Player 1's Move"
                            else:
                                self.label["text"] = f"Player 2's Move"

                        # needed for ai to take its turn
                        self.after(100, self.on_click)
            else:
                # AI's move
                move = self.game.ai_move()
                self.update_board()

                # add to moves made
                self.moves = self.moves + 1

                if self.game.is_game_over():
                    winner = self.game.get_winner()
                    self.announce_winner(winner)

                token_x, token_y = move['p1_removed_token']
                self.game.remove_token('P1', token_x, token_y)
                self.move_phase = True  # Switch back to pawn movement phase
                self.update_board()

                if self.game.is_game_over():
                    winner = self.game.get_winner()
                    self.announce_winner(winner)
                else:
                    self.label["text"] = f"Player 2's Move"

                # needed for ai to take its turn
                self.after(100, self.on_click)
        else:
            # AI's move
            move = self.game.ai_move()
            self.update_board()

            # add to moves made
            self.moves = self.moves + 1

            if self.game.is_game_over():
                winner = self.game.get_winner()
                self.announce_winner(winner)

            token_x, token_y = move['p2_removed_token']
            self.game.remove_token('P2', token_x, token_y)
            self.move_phase = True  # Switch to movement phase
            self.update_board()

            if self.game.is_game_over():
                winner = self.game.get_winner()
                self.announce_winner(winner)
            else:
                self.label["text"] = f"Player 1's Move"

            # needed for ai to take its turn
            self.after(100, self.on_click)

    def update_board(self):
        """
        Update visual board
        """
        for i in range(8):
            for j in range(6):
                text = self.game.board[i][j]
                self.buttons[i][j]["text"] = text

                # Tile colors
                if text == "P1":
                    self.buttons[i][j]["bg"] = "lightblue"
                elif text == "P2":
                    self.buttons[i][j]["bg"] = "red"
                elif text == "\n":
                    self.buttons[i][j]["bg"] = "white"
                else:
                    self.buttons[i][j]["bg"] = "black"
                    self.buttons[i][j]["state"] = "disabled"


def start_game():
    """
    Initiates game logic and gui logic and starts game
    """
    game = IsolationGame()
    app = IsolationApp(game)
    app.center_window()
    app.on_click()
    app.mainloop()


if __name__ == "__main__":
    start_game()
