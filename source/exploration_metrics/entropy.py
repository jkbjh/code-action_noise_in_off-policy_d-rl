from .density import KNN, suggest_knn_order
import numpy as np


def bi_nnr_relative_entropy_components(X, Y, k=None, C_L=None, C_U=None):
    Z = np.vstack([X, Y])
    MN, dim = Z.shape
    if k is None:
        k = int(np.maximum(np.sqrt(MN) * 0.2, 2))
    C_L = C_L if C_L is not None else 1 / (2 ** dim) * 1 / (MN)
    C_U = C_U if C_U is not None else 1 / C_L
    knn = KNN(Z)
    _, indices = knn.query(Z, k)
    Q_X = indices[: len(X)]
    Q_Y = indices[len(X) :]
    eps = np.finfo(np.float32).eps

    def g(t):
        return np.log(t)

    Q_X_X = np.sum(Q_X < len(X), axis=1)
    Q_Y_X = np.sum(Q_Y < len(X), axis=1)

    def nnr(X, Y, QX, QY):
        N = len(X)
        M = len(Y)
        Ni = QX
        Mi = QY
        eta = float(M) / float(N)
        x = (Ni + eps) / (Mi + 1.0) * eta
        hat_g = np.clip(g(x), g([C_L / C_U]), g([C_U / C_L]))
        return np.maximum(hat_g.mean(), 0.0)

    return nnr(X, Y, Q_X_X, k - Q_X_X), nnr(Y, X, k - Q_Y_X, Q_Y_X)


def bi_nnr_relative_entropy(X, Y, k=None, C_L=None, C_U=None):
    DKL_XY, DKL_YX = bi_nnr_relative_entropy_components(X, Y, k=k, C_L=C_L, C_U=C_U)
    return DKL_XY + DKL_YX
