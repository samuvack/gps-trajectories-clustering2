import similaritymeasures as simm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff


class FrechetDist:

    @staticmethod
    def dist(a, b):
        dists = np.linalg.norm(a - b, axis=1)
        return np.min(dists), np.argmin(dists)

    @staticmethod
    def dist_two_point(a, b):
        dist = np.linalg.norm(a - b)
        return dist

    @classmethod
    def dist_matrix(cls, P, Q):
        pn, qn = len(P), len(Q)
        dist_m = np.zeros((pn, qn))
        for i in range(pn):
            dist_m[i, :] = np.linalg.norm(Q - P[i], axis=1)
        return dist_m

    @classmethod
    def frechetdist_index(cls, P, Q):
        M = cls.dist_matrix(P, Q)
        pn, qn = len(P), len(Q)
        fd_m = np.zeros((pn, qn)) + np.infty
        fd_m[0, 0] = M[0, 0]
        for j in range(1, qn):
            fd_m[0, j] = max(fd_m[0, j - 1], M[0, j])
        for i in range(1, pn):
            fd_m[i, 0] = max(fd_m[i-1, 0], M[i, 0])
            for j in range(1, qn):
                r1 = fd_m[i-1, j-1]
                r2 = fd_m[i-1, j]
                r3 = fd_m[i, j-1]
                fd_m[i, j] = max(min(r1, r2, r3), M[i, j])
        _d = fd_m[pn-1, qn-1]
        _indices = np.where(fd_m == _d)
        _idx = np.argmin(_indices[0] + _indices[1])
        return _d, _indices[0][_idx], _indices[1][_idx]

    @classmethod
    def frechetdist_index_2(cls, P, Q):
        pn, qn = len(P), len(Q)
        if pn == qn == 1:
            dist = cls.dist_two_point(P[-1], Q[-1])
            return dist, 0, 0
        elif qn == 1:
            dists = np.linalg.norm(P - Q[-1], axis=1)
            d_max, idx_max = np.max(dists), np.argmax(dists)
            return d_max, idx_max, 0
        elif pn == 1:
            dists = np.linalg.norm(Q - P[-1], axis=1)
            d_max, idx_max = np.max(dists), np.argmax(dists)
            return d_max, 0, idx_max
        else:
            r1 = cls.frechetdist_index_2(P[:-1], Q[:-1])
            r2 = cls.frechetdist_index_2(P[:-1], Q)
            r3 = cls.frechetdist_index_2(P, Q[:-1])
            r = [r1, r2, r3]
            _idx_min = int(np.argmin([i[0] for i in r]))
            dist3 = cls.dist_two_point(P[-1], Q[-1])
            if r[_idx_min][0] > dist3:
                return r[_idx_min]
            return dist3, pn - 1, qn - 1

