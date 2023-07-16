"""
@author:  zhouxiao
@contact: zhouxiao17@mails.tsinghua.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# __all__ = ["AdaSPLoss"]

class AdaSPLoss(object):
    """
    Adaptive sparse pairwise (AdaSP) loss
    """

    def __init__(self, temp=0.04, is_train =False):
        self.temp = temp
        self.is_train = is_train

    def __call__(self, feats, targets):
        # print('adasp',feats.size())
        # print(feats)
        #为什么有些数据传入会是零位向量这可能和数据传入有问题
        #下面是adasp对传入的数据做的处理
        # if torch.numel(feats)==1:
        #     print('this is o demension tensor')
        #     feats_n=torch.zeros(8,3840).cuda()
        #     print(feats_n,"###################################")
        # else:
        

        feats_n = nn.functional.normalize(feats, dim=1)

        # print('feats_n',feats_n,feats_n.size()) 
        # feats_n = nn.functional.normalize(feats, dim=1)
        bs_size = feats_n.size(0)
        # N_id = len(torch.unique(targets))
        """N_ID
            原自适应识别可能对标签做了独特处理，不允许有重复标签出现。2023.7.10
            原自适应识别将此损失函数加入了mgn中对backbone、各个embeding_head都计算了损失
            待修改
        Returns:
            _type_: _description_
        """
        N_id=len(targets)
        N_ins = bs_size // N_id

        scale = 1./self.temp

        sim_qq = torch.matmul(feats_n, feats_n.T)
        sf_sim_qq = sim_qq*scale

        right_factor = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
        pos_mask = torch.from_numpy(np.kron(np.eye(N_id),np.ones((N_ins,1)))).cuda()
        left_factor = torch.from_numpy(np.kron(np.eye(N_id), np.ones((1,N_ins)))).cuda()
        
        ## hard-hard mining for pos
        mask_HH = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).cuda()
        # print(N_id)
        # print("N_ID:",np.eye(N_id))
        # print("N_ins",np.ones((N_ins,N_ins)))
        # print("mask_hh",mask_HH)
        
        mask_HH[mask_HH==0]=1.
        
        # print(mask_HH.shape)
        # print(sf_sim_qq.shape)
        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))#The size of tensor a (8) must match the size of tensor b (6) at non-singleton dimension 1
        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        pos_mask_id = torch.eye(N_id).cuda()
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH==0]=1.
        pos_sim_HH = 1./pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1-pos_mask_id) + pos_sim_HH.mul(pos_mask_id)
        
        # ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH,p = 1, dim = 1)   
        
        ## hard-easy mining for pos
        mask_HE = torch.from_numpy(np.kron(np.eye(N_id),-1.*np.ones((N_ins,N_ins)))).cuda()
        mask_HE[mask_HE==0]=1.

        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        pos_sim_HE[pos_sim_HE==0]=1.
        pos_sim_HE = 1./pos_sim_HE
        ID_sim_HE = ID_sim_HE.mul(1-pos_mask) + pos_sim_HE.mul(pos_mask)

        # hard-hard for neg
        ID_sim_HE = left_factor.mm(ID_sim_HE)

        # ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE,p = 1, dim = 1)
        
    
        l_sim = torch.log(torch.diag(ID_sim_HH))
        s_sim = torch.log(torch.diag(ID_sim_HE))

        weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach()/scale
        weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach()/scale
        wt_l = 2*weight_sim_HE.mul(weight_sim_HH)/(weight_sim_HH + weight_sim_HE)
        wt_l[weight_sim_HH < 0] = 0
        both_sim = l_sim.mul(wt_l) + s_sim.mul(1-wt_l) 
    
        adaptive_pos = torch.diag(torch.exp(both_sim))

        pos_mask_id = torch.eye(N_id).cuda()
        adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1-pos_mask_id)

        adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat,p = 1, dim = 1)

        # loss_sph = -1*torch.log(torch.diag(ID_sim_HH_L1)).mean()
        # loss_splh = -1*torch.log(torch.diag(ID_sim_HE_L1)).mean()
        loss_adasp = -1*torch.log(torch.diag(adaptive_sim_mat_L1)).mean()
        # print(loss_adasp)
        # if self.loss_type == 'sp-h':
        #     loss = loss_sph.mean()
        # elif self.loss_type == 'sp-lh':
        #     loss = loss_splh.mean()
        # elif self.loss_type == 'adasp':
        #     loss = loss_adasp
        return loss_adasp
# if __name__=="__main__":
    
#     a=torch.Tensor([[1,2,3],[1,2,3]])

#     ada=AdaSPLoss()
#     features = torch.rand(16, 2048)
#     targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
#     loss = ada(features, targets)
#     print(loss)
    
