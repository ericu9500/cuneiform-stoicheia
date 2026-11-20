"""Plane ids shared by the model, the noising and the (torch-free) baselines."""
PAD, MASK, BLANK, UNK_SIGN, X_SIGN, GAP = 0, 1, 2, 3, 4, 5
READ_NONE, READ_MASK = 0, 1
N_JOIN = 7            # w - . : + '' line
N_DET = 3
FIRST_REAL_SIGN = 6   # ids below are specials
