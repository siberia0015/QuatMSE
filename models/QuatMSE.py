import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class QuatMSE(Model):
    def __init__(self, config):
        super(QuatMSE, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.config.entTotal, 4 * self.config.hidden_size)  # vectorized quaternion
        self.rel_embeddings = nn.Embedding(self.config.relTotal, 4 * self.config.hidden_size)
        # self.Whr = nn.Embedding(self.config.relTotal, 4 * self.config.hidden_size)
        # self.Wtr = nn.Embedding(self.config.relTotal, 4 * self.config.hidden_size)
        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        # nn.init.xavier_uniform_(self.Whr.weight.data)
        # nn.init.xavier_uniform_(self.Wtr.weight.data)

    # 归一化
    @staticmethod
    def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
        size = quaternion.size(split_dim) // 4
        quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
        quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
        quaternion = quaternion.reshape(-1, 4 * size)
        return quaternion

    # 两个四元数向量之间的元素乘法
    @staticmethod
    def make_wise_quaternion(quaternion):  # for vector * vector quaternion element-wise multiplication
        if len(quaternion.size()) == 1:
            quaternion = quaternion.unsqueeze(0)
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
        i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
        j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
        k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
        return r2, i2, j2, k2

    @staticmethod
    def get_quaternion_wise_mul(quaternion):
        size = quaternion.size(1) // 4
        quaternion = quaternion.view(-1, 4, size)
        quaternion = torch.sum(quaternion, 1)
        return quaternion

    @staticmethod
    def vec_vec_wise_multiplication(q, p):  # vector * vector
        normalized_p = QuatMSE.normalization(p)  # bs x 4dim
        q_r, q_i, q_j, q_k = QuatMSE.make_wise_quaternion(q)  # bs x 4dim

        qp_r = QuatMSE.get_quaternion_wise_mul(q_r * normalized_p)  # qrpr−qipi−qjpj−qkpk
        qp_i = QuatMSE.get_quaternion_wise_mul(q_i * normalized_p)  # qipr+qrpi−qkpj+qjpk
        qp_j = QuatMSE.get_quaternion_wise_mul(q_j * normalized_p)  # qjpr+qkpi+qrpj−qipk
        qp_k = QuatMSE.get_quaternion_wise_mul(q_k * normalized_p)  # qkpr−qjpi+qipj+qrpk

        return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

    # TODO ((Vh*g(Vt,Vr))*Vr)x(Vt*g(Vh,Vr))
    @staticmethod
    def _calc(h, r, t, hr, tr):
        h_r = QuatMSE.vec_vec_wise_multiplication(h, hr) # Vh*Vr,1
        t_r = QuatMSE.vec_vec_wise_multiplication(t, tr) # Vt*Vr,2
        hrr = QuatMSE.vec_vec_wise_multiplication(h_r, r) # (Vh*Vr,1)*Vr
        hrt = hrr * t_r # ((Vh*Vr,1)*Vr)x(Vt*Vr,2)
        return -torch.sum(hrt, -1)

    # 正则化
    @staticmethod
    def regularization(quaternion):  # vectorized quaternion bs x 4dim
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

    def loss(self, score, regul):
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul

    @staticmethod
    def g1(v1, v2):
        return v1 * v2

    @staticmethod
    def g2(v1, v2):
        return QuatMSE.vec_vec_wise_multiplication(v1, v2)

    @staticmethod
    def g3(v1, v2):
        return v1 + v2

    @staticmethod
    def g4(v1, v2):
        return v1 - v2

    @staticmethod
    def g5(v1, v2):
        return nn.Sigmoid(QuatMSE.vec_vec_wise_multiplication(v1, v2))


    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        hr = QuatMSE.g4(t, r)
        tr = QuatMSE.g4(r, h)
        # hr = self.Whr(self.batch_r) # Vr.1
        # tr = self.Wtr(self.batch_r) # Vr.2

        score = QuatMSE._calc(h, r, t, hr, tr)
        regul = self.regularization(h) + self.regularization(r) + self.regularization(t) + self.regularization(hr) + self.regularization(tr)

        return self.loss(score, regul)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        hr = QuatMSE.g4(t, r)
        tr = QuatMSE.g4(r, h)
        # hr = self.Whr(self.batch_r)
        # tr = self.Wtr(self.batch_r)

        score = QuatMSE._calc(h, r, t, hr, tr)

        return score.cpu().data.numpy()
