import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from loss.adasp_loss import AdaSPLoss
from .backbones.crossvit import crossvit_base_224_TransReID


def shuffle_unit(features, shift, group, begin=1):#(shift=5,group=4)

    batchsize = features.size(0)
    # print(features.size())
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # print(x.shape)
    # Patch Shuffle Operation
    # try:
    #     x = x.view(batchsize, group, -1, dim)
    # except:
    #     x = torch.cat([x, x[:, -2:-1, :]], dim=1)
    #     x = x.view(batchsize, group, -1, dim)
    if x.size(1)%2==0:
        x = x.view(batchsize, group, -1, dim)
    else:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim) 
    # print(x.shape)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)
    # print(x.shape)
    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                print()
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        embed_dim=[384,768]
        self.cross_divide_length=cfg.MODEL.CROSS_LENGTH
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
            # print(camera_num)
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
            # print(view_num)
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        # print(block)
        # layer_norm = self.base.norm
        # self.b1 = nn.Sequential(
        #     copy.deepcopy(block),
        #     copy.deepcopy(layer_norm)
        # )
        # self.b2 = nn.Sequential(
        #     copy.deepcopy(block),
        #     copy.deepcopy(layer_norm)
        # )
        self.b1 = copy.deepcopy(block)
  
        self.b2 = copy.deepcopy(block)
         

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        # self.ada=cfg.MODEL.METRIC_LOSS_TYPE
        # self.tmp=cfg.MODEL.ADA_LOSS_WEIGHT
        self.eps=cfg.MODEL.LOSSES_CE_EPSILON
        self.alpha=cfg.MODEL.LOSSES_CE_ALPHA 
        self.scale=cfg.MODEL.LOSSES_CE_SCALE
        self.pixel_mean=cfg.INPUT.PIXEL_MEAN
        self.pixel_std=cfg.INPUT.PIXEL_STD
        
        self.norm = nn.ModuleList([nn.LayerNorm(embed_dim[i]) for i in range(2)])
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], self.in_planes) if self.in_planes > 0 else nn.Identity() for i in range(2)])
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        # print(x.shape)NOTE
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        
        # print(features.shape)#featrue[8,129,768]
        # global branch
        # b1_feat=features
        # print([features[i].shape for i in range(2)])
        # for blk in self.b1:
        #     b1_feat =blk(b1_feat)
        # b1_feat = [blk(features) for blk in self.b1] # [64, 129, 768]
        b1_feat=self.b1(features)
        # bh=b1_feat[0].size(0)
        # NOTE 2023.9.1
        b1_feat = [self.norm[i](x) for i, x in enumerate(b1_feat)]
        global_feat = [b1[:, 0] for b1 in b1_feat]
        global_feat=[self.head[i](x) for i ,x in enumerate(global_feat)]
        global_feat=torch.mean(torch.stack(global_feat,dim=0),dim=0)
        # global_feat=global_feat.view(bh,-1,self.in_planes)
                # ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        # ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        # JPM branch
        feature_length = [features[i].size(1) - 1 for i in range(2)]
        # print(feature_length)
        patch_length = [feature_length[i] // self.divide_length for i in range(2)]
        # token = features[:, 0:1]
        token=[]
        for i in range(2):
        # token0 = features[0][:, 0:1]
        # token1 = features[1][:, 0:1]
            token.append(features[i][:,0:1])
        x=[]
        if self.rearrange:
            for i in range(2):
                x.append(shuffle_unit(features[i], self.shift_num, self.shuffle_groups))
            # x0= shuffle_unit(features[0], self.shift_num, self.shuffle_groups) 
            # x1= shuffle_unit(features[1], self.shift_num, self.shuffle_groups) 
        else:
            for i in range(2):
                x.append(features[i][:,1:])
            # x0 = features[0][:, 1:]
            # x1 = features[1][:, 1:]
        # lf_1
        b1_local_feat_0 = x[0][:, :patch_length[0]]
        b1_local_feat_1 = x[1][:, :patch_length[1]]
        # print(b1_local_feat.size())
        # print(token0.size())
        b1_local_feat=[torch.cat((token[0],b1_local_feat_0),dim=1),torch.cat((token[1],b1_local_feat_1),dim=1)]
        b1_local_feat = self.b2(b1_local_feat)
        bh1=b1_local_feat[0].size(0)
        
        # NOTE 注意 
        b1_l_f= [self.norm[i](x) for i, x in enumerate(b1_local_feat)]
        b1_l_f = [b1[:, 0] for b1 in b1_local_feat]
        b1_l_f=[self.head[i](x) for i ,x in enumerate(b1_l_f)]
        b1_l_f=torch.mean(torch.stack(b1_l_f,dim=0),dim=0)
        b1_l_f=b1_l_f.view(bh1,-1,self.in_planes)
        local_feat_1 = b1_l_f[:, 0]
        # print(local_feat_1.size())
        # lf_2
        # b2_local_feat = x[1][:, patch_length[1]:patch_length[1]*2]
        # b2_local_feat = self.b2([torch.cat((token1, b2_local_feat), dim=1)])
        # local_feat_2 = b2_local_feat[:, 0]
        b2_local_feat_0 = x[0][:, patch_length[0]:patch_length[0]*2]
        b2_local_feat_1 = x[1][:, patch_length[1]:patch_length[1]*2]
        b2_local_feat=[torch.cat((token[0],b2_local_feat_0),dim=1),torch.cat((token[1],b2_local_feat_1),dim=1)]
        b2_local_feat = self.b2(b2_local_feat)
        bh2=b2_local_feat[0].size(0)
        b2_l_f= [self.norm[i](x) for i, x in enumerate(b2_local_feat)]
        b2_l_f = [b2[:, 0] for b2 in b2_local_feat]
        b2_l_f=[self.head[i](x) for i ,x in enumerate(b2_l_f)]
        b2_l_f=torch.mean(torch.stack(b2_l_f,dim=0),dim=0)
        b2_l_f=b2_l_f.view(bh2,-1,self.in_planes)
        local_feat_2 = b2_l_f[:, 0]

        # lf_3
        # b3_local_feat = x[:, patch_length[0]*2:patch_length[0]*3]
        # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        # local_feat_3 = b3_local_feat[:, 0]
        b3_local_feat_0 = x[0][:, patch_length[0]*2:patch_length[0]*3]
        b3_local_feat_1 = x[1][:, patch_length[1]*2:patch_length[1]*3]
        b3_local_feat=[torch.cat((token[0],b3_local_feat_0),dim=1),torch.cat((token[1],b3_local_feat_1),dim=1)]
        b3_local_feat = self.b2(b3_local_feat)
        bh3=b3_local_feat[0].size(0)
        b3_l_f= [self.norm[i](x) for i, x in enumerate(b3_local_feat)]
        b3_l_f = [b3[:, 0] for b3 in b3_local_feat]
        b3_l_f=[self.head[i](x) for i ,x in enumerate(b3_l_f)]
        b3_l_f=torch.mean(torch.stack(b3_l_f,dim=0),dim=0)
        b3_l_f=b3_l_f.view(bh3,-1,self.in_planes)
        local_feat_3 = b3_l_f[:, 0]
        # lf_4
        # b4_local_feat = x[:, patch_length[1]*3:patch_length[1]*4]
        # b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        # local_feat_4 = b4_local_feat[:, 0]
        b4_local_feat_0 = x[0][:, patch_length[0]*3:patch_length[0]*4]
        b4_local_feat_1 = x[1][:, patch_length[1]*3:patch_length[1]*4]
        b4_local_feat=[torch.cat((token[0],b4_local_feat_0),dim=1),torch.cat((token[1],b4_local_feat_1),dim=1)]
        b4_local_feat = self.b2(b4_local_feat)
        bh4=b4_local_feat[0].size(0)
        b4_l_f= [self.norm[i](x) for i, x in enumerate(b4_local_feat)]
        b4_l_f = [b2[:, 0] for b2 in b2_local_feat]
        b4_l_f=[self.head[i](x) for i ,x in enumerate(b4_l_f)]
        b4_l_f=torch.mean(torch.stack(b4_l_f,dim=0),dim=0)
        b4_l_f=b4_l_f.view(bh4,-1,self.in_planes)
        local_feat_4 = b4_l_f[:, 0]
        # print(global_feat.size())





        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            cls_score_1 = self.classifier_1(local_feat_1_bn)
            cls_score_2 = self.classifier_2(local_feat_2_bn)
            cls_score_3 = self.classifier_3(local_feat_3_bn)
            cls_score_4 = self.classifier_4(local_feat_4_bn)
            
            # cls_score=cross_entropy_loss(feat,label,self.eps,self.alpha)*self.scale*0.125
            # cls_score_1 = cross_entropy_loss(local_feat_1_bn,label,self.eps,self.alpha)*self.scale*0.125
            # cls_score_2 = cross_entropy_loss(local_feat_2_bn,label,self.eps,self.alpha)*self.scale*0.125
            # cls_score_3 = cross_entropy_loss(local_feat_3_bn,label,self.eps,self.alpha)*self.scale*0.125
            # cls_score_4 = cross_entropy_loss(local_feat_4_bn,label,self.eps,self.alpha)*self.scale*0.125
            
            # loss_ada=self.losses(global_feat,local_feat_1,local_feat_2,local_feat_3,local_feat_4,label)
            
            return torch.cat([cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4],dim=1), torch.cat([global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4],dim=1)  # global feature for triplet loss
           
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def preprocess_image(self,batch_input):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batch_input, dict):
            images = batch_input["images"].to(self.device)
        elif isinstance(batch_input, torch.Tensor):
            images = batch_input.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batch_input)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images
    def losses(self,
                    global_feat,
                    local_feat_1,local_feat_2,local_feat_3,local_feat_4,
                    label):
        """
        NOTE: 此函数由Adasp中meta_arch.mgn.losses改变
        Args:
            global_feat (_type_): _description_
            local_feat1 (_type_): _description_
            local_feat2 (_type_): _description_
            local_feat3 (_type_): _description_
            local_feat4 (_type_): _description_
        """
        loss_func=AdaSPLoss(is_train=True)
        feat=torch.cat((global_feat,local_feat_1,local_feat_2,local_feat_3,local_feat_4),dim=1)
        loss_ada=loss_func(feat,label)*self.tmp*0.2
    
        # print(feat_ada.shape)

        return loss_ada

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    "crossvit_base_224_TransReID":crossvit_base_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module===========')
            # print(model)
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
