import math
import torch
from torch import nn


class MultimodalLoss(nn.Module):
    def __init__(self, alpha, beta, delta, batch_size=16, device='cuda'): # 0.1 0.1 1.0
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        self.device = device
        self.CE_Fn = nn.CrossEntropyLoss()
        self.VA = torch.nn.SmoothL1Loss()
        # self.VA = torch.nn.MSELoss()



    def forward(self, vision_invariant_cls_out, audio_invariant_cls_out, text_invariant_cls_out, vision_specific_cls_out, audio_specific_cls_out, text_specific_cls_out, out_a, out_v, la , lv):
        batch_size = la.shape[0]
        vision_specific_label = torch.zeros((batch_size)).long().to(self.device)
        audio_specific_label = torch.ones((batch_size)).long().to(self.device)
        text_specific_label = torch.ones((batch_size)).long().to(self.device) * 2
        vision_invariant_label = torch.zeros((batch_size)).long().to(self.device)
        audio_invariant_label = torch.ones((batch_size)).long().to(self.device)
        text_invariant_label = torch.ones((batch_size)).long().to(self.device) * 2


        modality_specific_loss = self.CE_Fn(vision_specific_cls_out, vision_specific_label) + self.CE_Fn(audio_specific_cls_out, audio_specific_label) + self.CE_Fn(text_specific_cls_out, text_specific_label)  

        modality_invariant_loss = self.CE_Fn(vision_invariant_cls_out, vision_invariant_label) + self.CE_Fn(audio_invariant_cls_out, audio_invariant_label) + self.CE_Fn(text_invariant_cls_out, text_invariant_label) 
        
        # ClsLoss1 = self.VA(out_a, la)
        ClsLoss = self.VA(out_a, la)
        # ClsLoss = ClsLoss1 + ClsLoss2
        ClsLoss = ClsLoss

        loss =  self.alpha * modality_invariant_loss + self.beta * modality_specific_loss + self.delta * ClsLoss

        return loss, modality_invariant_loss, modality_specific_loss, ClsLoss


