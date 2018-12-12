"""
This program contains classes for building neural nets for HumanChess

Project: HumanChess
Path: root/net.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

import numpy as np

class NetError(Exception):
    """
    Basic exception for neural net.
    """
    pass

class UCIError(Exception):
    pass

class BaseNet(nn.Module):
    """
    Base class for neural nets. Inputs board state + a proposed move, outputs an affinity score.
    """
    def __init__(self):
        super().__init__()
        self.from_file = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
        self.to_file = dict()
        for file, ii in self.from_file.items():
            self.to_file[ii] = file
        # (8*8 board * 6 distinct pieces * 2 colors + 4 castling rights + 8 en_passant squares)
        # + 8*8 initial squares + 8*8 final squares = 908
        self.input_size = (8*8*6*2 + 4 + 8) + 8*8 + 8*8
        self.pieces = 'pnbrqkPNBRQK'
        self.layers = []
        self.activation_function = lambda x : x
    
    def fen_uci_lists_to_inputs(self, fen_list, uci_lists):
        uci_vector_list = []
        for uci_list in uci_lists:
            for uci in uci_list:
                uci_vector_list.append(self.uci_to_vector(uci))
        uci_array = np.array(uci_vector_list)

        fen_state_list = [self.fen_to_vector(fen) for fen in fen_list]
        fen_array = np.array(fen_state_list)

        move_counts = [len(uci_list) for uci_list in uci_lists]
        fen_array = np.repeat(fen_array, move_counts, axis = 0)
        
        inputs = np.concatenate((fen_array, uci_array), axis = 1)
        return torch.FloatTensor(inputs), move_counts

    def affinities_from_inputs(self, inputs):
        if len(self.layers) == 0:
            raise AttributeError('Undefined operation: compute_from_inputs')
        else:
            h = inputs
            for layer in self.layers[:-1]:
                h = self.activation_function(layer(h))
            y = self.layers[-1](h)
            return y
        return np.zeros(len(inputs))

    def forward(self, fen_list):
        # Find the legal moves for each fen
        uci_lists = [self.legal_ucis(fen) for fen in fen_list]
        return self.forward_select(fen_list, uci_lists)
    
    def forward_select(self, fen_list, uci_lists):
        #  Convert into inputs
        inputs, move_counts = self.fen_uci_lists_to_inputs(fen_list, uci_lists)
        # Calculate the affinities for each fen uci pair
        affinities = self.affinities_from_inputs(inputs)
        # Convert the affinities into probabilities
        probabilities = self.affinities_to_probabilities(affinities, move_counts)
        return uci_lists, probabilities
    
    def affinities_to_probabilities(self, affinities, move_counts):
        # Split the affinities
        split_affinities = torch.split(affinities[:,0], move_counts)
        # For every set of affinities apply softmax
        split_probabilities = [F.softmax(x) for x in split_affinities]
        return split_probabilities
    
    def fen_to_vector(self, fen):
        """
        Converts the FEN into a numpy array readable by the neural net

        FEN is of form rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6

        Returns:
            0 : board vector
            1 : castling vector
            2 : en_passant_square
        """
        board_string, to_play, castling, en_passant_string = fen.split(' ')

        board_vector = self.board_string_to_vector(board_string)

        castling_vector = np.zeros(4, dtype = np.bool_)
        if 'K' in castling:
            castling_vector[0] = 1
        if 'Q' in castling:
            castling_vector[1] = 1
        if 'k' in castling:
            castling_vector[2] = 1
        if 'q' in castling:
            castling_vector[3] = 1
        
        en_passant = np.zeros(8, dtype=np.int_)
        for file, ii in self.from_file.items():
            if file in en_passant_string:
                en_passant[ii] = 1
        
        fen_vector = np.concatenate((board_vector, castling_vector, en_passant))
        return fen_vector

    def uci_to_vector(self, uci):
        uci2d0 = np.zeros((8,8), dtype=np.bool_)
        uci2d1 = np.zeros((8,8), dtype=np.bool_)
        try:
            file0 = self.from_file[uci[0]]
            rank0 = int(uci[1])-1
            file1 = self.from_file[uci[2]]
            rank1 = int(uci[3])-1

            uci2d0[file0,rank0] = 1
            uci2d1[file1,rank1] = 1
        except UCIError as e:
            raise UCIError('Error with uci %s' % (uci,))
        uci0 = uci2d0.flatten()
        uci1 = uci2d1.flatten()
        uci_vector = np.concatenate((uci0, uci1), axis=0)
        return uci_vector

    def uci(self, vector):
        uci0, uci1 = vector[:64], vector[64:]
        uci2d0 = np.reshape(uci0, (8,8))
        uci2d1 = np.reshape(uci1, (8,8))
        indeces0 = np.argwhere(uci2d0)[0]
        indeces1 = np.argwhere(uci2d1)[0]
        file0 = self.to_file[indeces0[0]]
        rank0 = str(indeces0[1]+1)
        file1 = self.to_file[indeces1[0]]
        rank1 = str(indeces1[1]+1)
        return file0 + rank0 + file1 + rank1

    def legal_ucis(self, fen):
        full_fen = fen + ' 10 66' # Add on dummy values for the halfclock moves and fullclock number
        board = chess.Board(full_fen)
        ucis = [x.uci() for x in board.legal_moves]
        return ucis

    def board_string_to_vector(self, board_string):
        vec = np.zeros((8,8,6*2), dtype=np.bool_)
        with_spaces = board_string.replace('1',' ')
        with_spaces =  with_spaces.replace('2','  ')
        with_spaces =  with_spaces.replace('3','   ')
        with_spaces =  with_spaces.replace('4','    ')
        with_spaces =  with_spaces.replace('5','     ')
        with_spaces =  with_spaces.replace('6','      ')
        with_spaces =  with_spaces.replace('7','       ')
        with_spaces =  with_spaces.replace('8','        ')
        ranks = with_spaces.split('/')
        for i_rank, rank in enumerate(reversed(ranks)):
            for i_file, occupant_str in enumerate(rank):
                if occupant_str == ' ':
                    pass
                else:
                    occupant_index = self.pieces.index(occupant_str)
                    vec[i_file, i_rank, occupant_index] = 1
        return vec.flatten()


class Net0(BaseNet):
    """
    First try at a net. Linear model.
    """
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(self.input_size, 1)]

class Trainer:

    def __init__(self):
        dataset = None
        training_set = None
        validation_set = None
        self.test_set = None
        self.batch_size = 0
        self.lr_scheduler = None
    
    def load_data(self):
        pass
    
    def partition_data(self):
        pass

    def loss_criterion(explorer, uci_lists, probabilities):
        """
        This calculates the loss with the Cross entropy loss over all the possibilities.

        Arguments:
            explorer dictionary : Moves in the database
            uci_lists : List of lists of examined moves
            probabilities : List of lists of probabilities for taking any of the examined moves.
        """



        Q_pi_vec = Q_pi.view((-1,1)).repeat(1,len(gl.WASD))
        delta = torch.add(Q, torch.neg(Q_pi_vec))

        flipper = torch.ones(delta.size(), dtype=torch.float)
        for i, a_val in enumerate(a):
            if delta[i,a_val] < 0:
                flipper[i,a_val] = -1

        delta = torch.mul(flipper, delta)

        delta = torch.clamp(delta, min=0.0)

        loss = torch.sum(delta, dim=1)
        return loss
net = Net0()
uciss, probss = net.forward(['8/5p2/5k2/2P5/2K5/8/8/8 w - -', '7k/7P/8/8/8/8/p7/K7 w - -'])
for ucis, probs in zip(uciss, probss):
    print('hey')
    for uci, prob in zip(ucis, probs):
        print(uci, '|', float(prob))