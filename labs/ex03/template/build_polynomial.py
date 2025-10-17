# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    N = x.shape[0]
    poly_matrix = np.zeros((N, degree + 1))  # Initialise la matrice finale

    for j in range(degree + 1):
        poly_matrix[:, j] = x ** j  # Remplit la colonne j avec x^j

    return poly_matrix
