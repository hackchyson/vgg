from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def cos_sim(mat1, mat2):
    return cosine_similarity(mat1.reshape(1, -1), mat2.reshape(1, -1))


# def max_sim(mat1, lst):
#     for i in range(len(lst)):
#         lst[i] = cos_sim(mat1, lst[i])
#     return max(lst)

def max_sim(mat1, lst):
    tmp_max = cos_sim(mat1, lst[0])
    print('info: idx = 0', tmp_max)
    max_idx = 0
    for i in range(1, len(lst)):
        cur = cos_sim(mat1, lst[i])
        print('info: idx = {}'.format(i), cur)
        if tmp_max < cur:
            tmp_max = cur
            max_idx = i
    return max_idx


# mat = np.array([1, 2, 3])
# lst = [[1, 2, 2.5], [2, 2, 3], [1, 3, 2]]
# lst = np.array(lst)
# print(type(lst))
#
# idx = max_sim(mat, lst)
# print('max index: ', idx)
#
# dd = np.dot(mat, np.transpose(lst[0]))
# ddd = np.linalg.norm(mat) * np.linalg.norm(lst[0])
# print(dd, ddd)
# print(dd/ddd)
