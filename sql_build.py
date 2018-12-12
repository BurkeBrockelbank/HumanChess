
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
import json
import urllib.request

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
        self.json_walk = self.json_walk_looping

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
                Time_Format TEXT,
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
    
    def add_move(self, position_ID, uci, popularity = -1):
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
        # Now do an increment if popularity is not specified
        if popularity == -1:
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
        # If popularity is specified, then update the popularity
        else:
            self.c.execute("""
                UPDATE
                    Move
                SET
                    Popularity = ?
                WHERE
                    Position_ID = ?
                AND
                    UCI = ?
            ;""", (popularity, position_ID, uci))

    def flip_fen(self, fen):
        """
            Flips a FEN with black to move into a mirrored FEN with white to move or vise versa
        """
        # Split the FEN
        fen_split = fen.split(' ')

        # Make sure it really is black to play
        white_to_play = fen_split[1] == 'w'
        fen_split[1] = ['w', 'b'][white_to_play]

        # Deal with the en passant part
        en_passant = fen_split[-1]
        if en_passant != '-':
            old_rank, new_rank = ['6', '3'][::white_to_play]
            en_passant = en_passant.replace(old_rank, new_rank)
            fen_split[-1] = en_passant
        
        # Deal with the castling rights
        castling = fen_split[2]
        if castling != '-':
            # Someone has castling rights
            new_castling = ''
            if 'k' in castling:
                new_castling = 'K'
            if 'q' in castling:
                new_castling += 'Q'
            if 'K' in castling:
                new_castling += 'k'
            if 'Q' in castling:
                new_castling += 'q'
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
            Flips a move made by black into the mirrored move made by white or vise versa
        """
        flipper = {'1':'8', '2':'7', '3':'6', '4':'5', '5':'4', '6':'3', '7':'2', '8':'1'}
        new_uci = uci[0] + flipper[uci[1]] + uci[2] + flipper[uci[3]]
        return new_uci

    def _json_moves(self, position_ID, json_string):
        json_moves = json.loads(json_string)['moves']
        move_popularity = list()
        for move in json_moves:
            uci = move['uci']
            popularity = move['white'] + move['draws'] + move['black']
            move_popularity.append((uci, popularity))
        return move_popularity

    def add_json(self, position_ID, move_max = -1,
        speeds = ['bullet', 'blitz', 'rapid', 'classical'],
        ratings = [1600, 1800, 2000, 2200, 2500],
        time_format = None,
        rating_pool = None):
        """
        This method starts from position at position_ID and adds the moves and resultant positions to the database
        according to the lichess opening explorer database.
        """
        # Look up the fen
        self.c.execute("""
            SELECT
                FEN
            FROM
                Position
            WHERE
                Position_ID = ?
        ;""", (position_ID, ))
        white_fen = self.c.fetchone()[0] + ' 0 0' # white to move
        # Get the flipped fen to check for the same position played as black
        black_fen = self.flip_fen(white_fen) # black to move
        # Build the url to the json
        url = 'https://explorer.lichess.ovh/lichess?variant=standard'
        for speed in speeds:
            url += '&speeds[]=%s' % (speed,)
        for rating in ratings:
            url += '&ratings[]=%d' % (rating,)
        if move_max != -1:
            url += '&moves=%d' % (move_max,)
        url += '&topGames=0&recentGames=0'
        white_url = url + '&fen=%s' % (white_fen.replace(' ', '%20'),)
        black_url = url + '&fen=%s' % (black_fen.replace(' ', '%20'),)
        # Retrieve the json file and get the moves
        with urllib.request.urlopen(white_url) as json_file:
            white_moves = self._json_moves(position_ID, json_file.read())
        with urllib.request.urlopen(black_url) as json_file:
            black_moves = self._json_moves(position_ID, json_file.read())
        # Now consolidate the moves together by making them both black moves.
        black_moves_combined = dict()
        for uci, popularity in white_moves:
            black_moves_combined[self.flip_uci(uci)] = popularity
        for uci, popularity in black_moves:
            if uci in black_moves_combined.keys():
                black_moves_combined[uci] += popularity
            else:
                black_moves_combined[uci] = popularity
        # Add moves to database
        for uci, popularity in black_moves_combined.items():
            uci_white = self.flip_uci(uci)
            self.add_move(position_ID, uci_white, popularity)
        # Get the list of moves to iterate over
        ucis = black_moves_combined.keys()
        # Add each resultant positions to database
        new_position_IDs = []
        for uci in ucis:
            board = chess.Board(black_fen)
            board.push_uci(uci)
            new_fen = board.fen()
            # Ignore halfclock and fullclock counts
            fen_strip = ' '.join(new_fen.split(' ')[:-2])
            new_ID = self.add_position(time_format, rating_pool, fen_strip)
            new_position_IDs.append(new_ID)
        # Return useful numbers
        if new_position_IDs == []:
            1 == 1
        return new_position_IDs
    
    def json_walk_recursive(self, position_ID, move_max = -1, max_depth = -1, depth = 0,
        speeds = ['bullet', 'blitz', 'rapid', 'classical'],
        ratings = [1600, 1800, 2000, 2200, 2500],
        time_format = None,
        rating_pool = None):
        """
        This method starts from position at position_ID and walks through lichess' online
        json library for every possible move up to move_count to a certain depth.
        """
        print('Walking from position %d as depth %d' % (position_ID, depth))
        # Recursive algorithm
        if depth != max_depth:
            new_position_IDs = self.add_json(position_ID, move_max, speeds, ratings, time_format, rating_pool)
            print('Created new positions', new_position_IDs)
            for pos_ID in new_position_IDs:
                self.json_walk(pos_ID, move_max, max_depth, depth + 1, speeds, ratings, time_format, rating_pool)
    
    def json_walk_looping(self, position_ID, move_max = 3, max_depth = 3,
        speeds = ['bullet', 'blitz', 'rapid', 'classical'],
        ratings = [1600, 1800, 2000, 2200, 2500],
        time_format = None,
        rating_pool = None):
        """
        This method starts from position at position_ID and walks through lichess' online
        json library for every possible move up to move_count to a certain depth.
        """
        # Figure out how many tasks there will be
        tasks = move_max**max_depth //(move_max-1)
        tasks_complete = 0
        # Make progress bar
        bar = progressbar.ProgressBar(max_value = tasks)
        # Keep a heap of all tasks to do.
        to_check = [(position_ID, 0)]
        while len(to_check) > 0:
            # Take the highest depth one
            check_ID, check_depth = to_check.pop(-1)
            # Calculate one more level of depth
            new_position_IDs = self.add_json(position_ID, move_max, speeds, ratings, time_format, rating_pool)
            # Add to the heap if the depth is not too high
            if check_depth + 1 != max_depth:
                for new_ID in new_position_IDs:
                    to_check.append((new_ID, check_depth+1))
            # Another task complete
            tasks_complete += 1
            bar.update(tasks_complete)
        bar.finish()

            



db = Chess_DB('lichess_databases/sql/json_walk.sqlite')
db.delete()
db.open()
db.initialize()
db.add_position('classical', 1600, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -')
# db.add_json(1,3,speeds=['classical'], ratings = [1600], time_format='classica', rating_pool=1600)
db.json_walk(1,3,10, speeds=['classical'], ratings = [1600], time_format='classical', rating_pool=1600)
db.commit()
db.close()