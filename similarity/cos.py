from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def cos_sim(mat1, mat2):
    """
    Compute the cosine similarity of the mat1 and mat2.

    :param mat1: A 2-D matrix that representing a image.
    :param mat2: A 2-D matrix that representing a image.

    :return: The cosine similarity of mat1 and mat2.
    """
    return cosine_similarity(mat1.reshape(1, -1), mat2.reshape(1, -1))


def max_sim(mat, lst):
    """
    Get the maximum cosine similarity in lst with mat.

    :param mat: The target matrix that needs be most similar to.
    :param lst: The candidates in which to find the most similar to mat.

    :return: the most similar matrix index in lst and maximum cosine similarity
    """
    tmp_max = cos_sim(mat, lst[0])
    # print('info: idx = 0', tmp_max)

    max_idx = 0
    for i in range(1, len(lst)):
        cur = cos_sim(mat, lst[i])
        print('info: idx = {}'.format(i), cur)
        if tmp_max < cur:
            tmp_max = cur
            max_idx = i
    return max_idx, cos_sim(mat, lst[max_idx])


