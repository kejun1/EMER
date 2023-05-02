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
        self.IN_Fn = torch.nn.SmoothL1Loss()



    def forward(self, vision_invariant_cls_out, audio_invariant_cls_out, text_invariant_cls_out, vision_specific_cls_out, audio_specific_cls_out, text_specific_cls_out, emotion_cls_output, in_output, cls_label, in_label):
        batch_size = cls_label.shape[0]
        vision_specific_label = torch.zeros((batch_size)).long().to(self.device)
        audio_specific_label = torch.ones((batch_size)).long().to(self.device)
        text_specific_label = torch.ones((batch_size)).long().to(self.device) * 2
        vision_invariant_label = torch.zeros((batch_size)).long().to(self.device)
        audio_invariant_label = torch.ones((batch_size)).long().to(self.device)
        text_invariant_label = torch.ones((batch_size)).long().to(self.device) * 2


        modality_specific_loss = self.CE_Fn(vision_specific_cls_out, vision_specific_label) + self.CE_Fn(audio_specific_cls_out, audio_specific_label) + self.CE_Fn(text_specific_cls_out, text_specific_label)  

        modality_invariant_loss = self.CE_Fn(vision_invariant_cls_out, vision_invariant_label) + self.CE_Fn(audio_invariant_cls_out, audio_invariant_label) + self.CE_Fn(text_invariant_cls_out, text_invariant_label) 
        
        ClsLoss = self.CE_Fn(emotion_cls_output, cls_label)
        INLoss = self.IN_Fn(in_output, in_label)
        total_loss = ClsLoss + INLoss

        loss =  self.alpha * modality_invariant_loss + self.beta * modality_specific_loss + self.delta * total_loss

        return loss, modality_invariant_loss, modality_specific_loss, ClsLoss


