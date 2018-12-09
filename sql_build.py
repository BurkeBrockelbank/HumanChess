
"""
This program builds a sql database for training the neural net.

All games are recorded from the side of the player to move. If that player is white, the FEN is unaltered but if that player is black, the colors are flipped.
Thus, all games appear to be white to play.

TABLE Position
Position_ID | Time_Format | Rating_Pool | FEN

Table Move
Move_ID | Position_ID | Popularity | UCI

Position_ID: Identification number
Time_Format: Either 0 or 1 or 2[integer]. 0 is short (Initial time + increment * 40 < 20 minutes), 1 is long (Initial time + increment * 40 >= 20 minutes), 2 is correspondence
Rating_Pool: The player's rating rounded down to the nearest hundred [integer].
FEN_pos: E.g. rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 [string]. Should always be white (capital letters) to move. Halfclock count and fullmove number are ignored.
k: Black's kingside castling rights [boolean].
Popularity: The number of players that made this move
Move: e.g. e1g1 [string]. Takes the form of initial square then final square.

Project: HumanChess
Path: root/sql_build.py
"""

import chess.pgn
import sqlite3

import shutil
import os

import progressbar

class TagError(Exception):
    pass

class Chess_DB:
    """
    Database object
    """

    def __init__(self, path):
        self.path = path

    def open(self):
        self.conn = sqlite3.connect(self.path)
        self.c = self.conn.cursor()

    def new_copy(self, copy_path):
        # See if the database is already open
        try:
            self.conn.close()
        except AttributeError:
            pass
        # Copy the database to a new location
        shutil.copy(self.path, copy_path)
        # Update self.path
        self.path = copy_path

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def delete(self):
        os.remove(self.path)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def initialize(self):
        """
            Create empty tables if they don't already exist.
        """
        # Position
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS Position (
                Position_ID INTEGER PRIMARY KEY,
                Time_Format INTEGER,
                Rating_Pool INTEGER,
                FEN TEXT
            );
        """)
        self.c.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
                Position_Time_Format_Rating_Pool_FEN 
            ON
                Position (
                    Time_Format,
                    Rating_Pool,
                    FEN
                )
        ;""")

        # Moves
        self.c.execute("""
            CREATE TABLE IF NOT EXISTS Move (
                Move_ID INTEGER PRIMARY KEY,
                Position_ID INTEGER,
                Popularity INTEGER,
                UCI TEXT
            );
        """)
        self.c.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
                Position_ID_UCI
            ON
                Move (
                    Position_ID, UCI
                )
        ;""")

    def add_pgn(self, pgn_path):
        """
            Adds all the games from the pgn to the database.
        """
        # Estimate the number of games
        # Get the file size of the pgn
        pgn_size = os.path.getsize(pgn_path) #bytes
        normal_game_size = 2133 #bytes
        n_games_estimate = pgn_size//normal_game_size


        widgets = ['Reading games: ', progressbar.Counter(), '/', str(n_games_estimate), ' (approximate)  ', progressbar.Timer()]
        bar = progressbar.ProgressBar(max_value = progressbar.UnknownLength, widgets=widgets)
        # Count number of games
        n_bad_games = 0
        n_games = 0
        with open(pgn_path, 'r') as pgn:
            game = chess.pgn.read_game(pgn)
            while game is not None:
                n_games += 1
                try:
                    self.add_game(game)
                except TagError as e:
                    print('TagError on game %d:' %(i,), str(e))
                    print(game)
                    n_bad_games += 1
                if n_games % 10 == 0:
                    bar.update(n_games)
                game = chess.pgn.read_game(pgn)
        bar.finish()

        print('Analysed %d/%d games' % (n_games-n_bad_games, n_games))

    def add_game(self, game):
        """
        Builds database from a single game passed as a chess.pgn.Game object, e.g.
        [Event "Rated Bullet game"]
        [Site "https://lichess.org/is641xjg"]
        [Date "2018.10.31"]
        [Round "-"]
        [White "Ryanbulo"]
        [Black "ciociaro"]
        [Result "1-0"]
        [UTCDate "2018.10.31"]
        [UTCTime "23:00:00"]
        [WhiteElo "1629"]
        [BlackElo "1576"]
        [WhiteRatingDiff "+9"]
        [BlackRatingDiff "-8"]
        [ECO "A00"]
        [Opening "Hungarian Opening: Catalan Formation"]
        [TimeControl "60+0"]
        [Termination "Normal"]

        1. g3 { [%clk 0:01:00] } d5 { [%clk 0:01:00] } 2. Bg2 { [%clk 0:01:00] } e6 { [%clk 0:01:00] } --> 55. Qa6 { [%clk 0:00:00] } Kc8 { [%clk 0:00:06] } 56. Qa8# { [%clk 0:00:00] } 1-0
        """

        # Get the important information out that is constant over every move
        try:   
            white_elo = int(game.headers['WhiteElo'])
            black_elo = int(game.headers['BlackElo'])
            white_rating_pool = int(white_elo/100)*100
            black_rating_pool = int(black_elo/100)*100

            time_control_str = game.headers['TimeControl']
            if '-' in time_control_str:
                time_format = 2
            else:
                time_control = time_control_str.split('+')
                initial_time = int(time_control[0])
                increment = int(time_control[1])
                normal_game_time = (initial_time + 40 * increment)/60
                if normal_game_time >=20:
                    time_format = 1
                elif normal_game_time < 20:
                    time_format = 0
        except ValueError as e:
            raise TagError(e)
        
        # Iterate through the moves and add the position in each instance
        board = game.board()
        for move in game.main_line():
            fen_split = board.fen().split(' ')
            # Ignore the fifty move rule and the move count
            fen = ' '.join(fen_split[:4])
            # Get the move in string format
            uci = move.uci()
            # Flip the fen and uci and rating pool if it's black to play
            rating_pool = white_rating_pool
            if fen_split[1] == 'b':
                fen = self.flip_fen(fen)
                uci = self.flip_uci(uci)
                rating_pool = black_rating_pool
            # Add to the database
            self.add_state(time_format, rating_pool, fen, uci)
            # Do the move
            board.push(move)


    def add_state(self, time_format, rating_pool, fen, uci):
        """
            Add a game state to the Position and Move tables
        """
        # Add to Position table
        position_ID = self.add_position(time_format, rating_pool, fen)
        # Add to Move table
        self.add_move(position_ID, uci)
    
    def add_position(self, time_format, rating_pool, fen):
        """
            Add a position to the Position table
        """
        # Add or ignore this position
        self.c.execute("""
            INSERT OR IGNORE INTO Position (
                Time_Format,
                Rating_Pool,
                FEN
            )
            VALUES (?, ?, ?)
        ;""", (time_format, rating_pool, fen))
        # Get the index from the position_table
        self.c.execute('SELECT Position_ID FROM Position WHERE Time_Format = ? AND Rating_Pool = ? AND FEN = ?', (time_format, rating_pool, fen))
        position_ID = self.c.fetchone()[0]
        # return this index
        return position_ID
    
    def add_move(self, position_ID, uci):
        """
            Add a uci move to the Move table.
        """
        # First do an insert or ignore
        self.c.execute("""
            INSERT OR IGNORE INTO Move (
                Position_ID,
                Popularity,
                UCI
            )
            VALUES (?, 0, ?)
        ;""", (position_ID, uci))
        # Now do an increment
        self.c.execute("""
            UPDATE
                Move
            SET
                Popularity = Popularity + 1
            WHERE
                Position_ID = ?
            AND
                UCI = ?
        ;""", (position_ID, uci))
        
    def flip_fen(self, fen):
        """
            Flips a FEN with black to move into a mirrored FEN with white to move.
        """
        # Split the FEN
        fen_split = fen.split(' ')

        # Make sure it really is black to play
        fen_split[1] = 'w'

        # Deal with the en passant part
        en_passant = fen_split[-1]
        if en_passant != '-':
            en_passant = en_passant.replace('3','6')
            fen_split[-1] = en_passant
        
        # Deal with the castling rights
        castling = fen_split[2]
        if castling != '-':
            # Someone has castling rights
            new_castling = ''
            if 'k' in castling:
                new_castling = 'K'
            if 'q' in castling:
                new_castling = 'Q'
            if 'K' in castling:
                new_castling = 'k'
            if 'Q' in castling:
                new_castling = 'q'
            fen_split[2] = new_castling

        # Deal with board diagram
        board = fen_split[0]
        # First swap the cases of the string
        board = board.swapcase()
        # Now we need to swap the first and eigths ranks, second and seventh etc.
        board_split = board.split('/')
        board_split.reverse()
        board = '/'.join(board_split)
        fen_split[0] = board

        # Rebuild the FEN
        return ' '.join(fen_split)

    def flip_uci(self, uci):
        """
            Flips a move made by black into the mirrored move made by white.
        """
        flipper = {'1':'8', '2':'7', '3':'6', '4':'5', '5':'4', '6':'3', '7':'2', '8':'1'}
        new_uci = uci[0] + flipper[uci[1]] + uci[2] + flipper[uci[3]]
        return new_uci

# db = Chess_DB('lichess_databases/sql/large_test.sqlite')
# db.open()
# db.initialize()
# db.add_pgn('lichess_databases/test/large.pgn')
# db.commit()
# db.close()