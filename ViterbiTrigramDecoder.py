from Key import Key
import math
import sys
import numpy as np
import codecs
import argparse

"""
This file is part of the computer assignments for the course DD1418 at KTH.
"""

class ViterbiTrigramDecoder(object):
    """
    This class implements Viterbi decoding using trigram probabilities in order
    to correct keystroke errors.
    """
    def init_a(self, filename):
        """
        Reads the trigram probabilities (the 'A' matrix) from a file.
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                i, j, k, d = [func(x) for func, x in zip([int, int, int, float], line.strip().split(' '))]
                self.a[i][j][k] = d


    # ------------------------------------------------------


    def init_b(self):
        """
        Initializes the observation probabilities (the 'B' matrix).
        """
        for i in range(Key.NUMBER_OF_CHARS):
            cs = Key.neighbour[i]

            # Initialize all log-probabilities to some small value.
            for j in range(Key.NUMBER_OF_CHARS):
                self.b[i][j] = -float("inf")

            # All neighbouring keys are assigned the probability 0.1
            for j in range(len(cs)):
                self.b[i][Key.char_to_index(cs[j])] = math.log( 0.1 )

            # The remainder of the probability mass is given to the correct key.
            self.b[i][i] = np.log((10 - len(cs))/10.0)


    # ------------------------------------------------------



    def viterbi(self, s):
        """
        Performs the Viterbi decoding and returns the most likely
        string.
        """
        # First turn chars to integers, so that 'a' is represented by 0,
        # 'b' by 1, and so on.
        index = [Key.char_to_index(x) for x in s]

        # The Viterbi matrices
        self.v = np.zeros((len(s), Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double')
        self.v[:,:,:] = -float("inf")
        self.backptr = np.zeros((len(s), Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='int')

        # Initialization

        # Paret: första bokstaven + inget (start), ska peka på start
        self.backptr[0,Key.START_END,:] = Key.START_END
        
        self.v[0,Key.START_END,:] = self.a[Key.START_END, Key.START_END, :] + self.b[index[0], :]

        # Induction step

        for t in range(1, len(s)):
            for j in range(Key.NUMBER_OF_CHARS):
                for k in range(Key.NUMBER_OF_CHARS):

                    # Förra viterbi eftersom v[t, j, k] är viterbi
                    # vid tiden t med paret j, k hittar vi nu alla förra t-1
                    # par :, j
                    prev_viterbi_prob = self.v[t-1,:,j]
                    # Nästa bokstav probability
                    next_char_prob = self.a[:, j, k]

                    # Added prob
                    prob = prev_viterbi_prob + next_char_prob

                    # Bästa förra viterbi
                    best_previous_prob = np.max(prob)

                    # Index av max, bokstaven i som ledde till j. k
                    letter_index_highest_score = np.argmax(prob)


                    # Prob att man tryckte rätt
                    intentional_press_prob = self.b[index[t], k]

                    # Nya viterbi value
                    self.v[t, j, k] = best_previous_prob + intentional_press_prob

                    # Spara bästa föregående bokstaven i (paret i, j) för att nå till j, k
                    self.backptr[t, j, k] = letter_index_highest_score


        # Finally return the result

        reversed_answer = ""
        j = Key.START_END
        k = Key.START_END

        for t in range(1, len(s)-2): # Kör loopen len(s) - 3 ggr börja på 1 för att hoppa över slutet [START_END, START_END]
            # Hämta bokstavsindex från backptr
            next_backptr = self.backptr[len(s)-1-t, j, k]

            # Index till bokstav och lägg till i strängen
            reversed_answer += Key.index_to_char(next_backptr)

            # Uppdatera backptr
            k = j
            j = next_backptr
            

        return reversed_answer[::-1]



    # ------------------------------------------------------



    def __init__(self, filename=None):
        """
        Constructor: Initializes the A and B matrices.
        """
        # The trellis used for Viterbi decoding. The first index is the time step.
        self.v = None

        # The trigram stats.
        self.a = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double')

        # The observation matrix.
        self.b = np.zeros((Key.NUMBER_OF_CHARS, Key.NUMBER_OF_CHARS), dtype='double')

        # Pointers to retrieve the topmost hypothesis.
        backptr = None

        if filename: self.init_a(filename)
        self.init_b()



    # ------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='ViterbiTrigram decoder')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', type=str, help='decode the contents of a file')
    group.add_argument('--string', '-s', type=str, help='decode a string')
    parser.add_argument('--probs', '-p', type=str,  required=True, help='trigram probabilities file')

    arguments = parser.parse_args()

    if arguments.file:
        with codecs.open(arguments.file, 'r', 'utf-8') as f:
            s1 = f.read().replace('\n', ' ')
    elif arguments.string:
        s1 = arguments.string
    s1 = s1.replace('\r', '')


    # Give the filename of the trigram probabilities as a command line argument
    d = ViterbiTrigramDecoder(arguments.probs)

    # Append two extra "END" symbols to the input string, to indicate end of sentence. 
    result = d.viterbi(s1 + Key.index_to_char(Key.START_END) + Key.index_to_char(Key.START_END))

    print(result)

if __name__ == "__main__":
    main()
