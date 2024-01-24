# import torchsummary

import torch
from torch import nn
from einops import rearrange
from model.MAQT.transformer import Transformer, CrossTransformer
import random
from torch.autograd import Function
# from torchvison.model import resnet
from model.resnet import resnet18
from einops import rearrange
import torch.autograd as autograd

class ResNet_LSTM_extract(nn.Module):
    def __init__(self, e_input_dim, v_input_dim, hidden_dim, n_layer, n_classes, pretrained=True):
        super(ResNet_LSTM_extract, self).__init__()
        self.f_extractor = resnet18(pretrained=pretrained)
        self.e_lstm = nn.LSTM(e_input_dim, hidden_dim,
                            n_layer, batch_first=True)  # dropout=0.5
        self.v_lstm = nn.LSTM(v_input_dim, hidden_dim,
                              n_layer, batch_first=True)
        self.classifier = nn.Linear(768, n_classes)

    # face:(8,8,3,112,112)  eye:(8,32,39) video:(8,32,2)
    def forward(self, face, eye ,video):
        face_seq_len = face.shape[1]#8
        x_f = rearrange(face, 'b f c h w -> (b f) c h w')#(64,3,112,112)
        x_f = self.f_extractor(x_f)  # (64,512)
        x_f = rearrange(x_f, '(b f) c -> b f c', f=face_seq_len)#(16,8,512)
        # x_f = x_f.mean(dim=1)  # (8,512)
        e_out, (e_h_n, e_c_n) = self.e_lstm(eye)#out:(16,32,128) 
        v_out, (v_h_n, v_c_n) = self.v_lstm(video)#out:(16,32,128) 
        x_e = e_out  # x:(16,32,128) 
        x_v = v_out # x:(16,32,128) 
        return x_f, x_e, x_v

class GradReverse(torch.autograd.Function):

    # 重写父类方法的时候，最好添加默认参数，不然会有warning（为了好看。。）
    @ staticmethod
    def forward(ctx, x, constant):
        #　其实就是传入dict{'lambd' = lambd}
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 传入的是tuple，我们只需要第一个
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class MAQT(nn.Module):
    def __init__(self, num_classs, transformer_depth=2):
        super(MAQT, self).__init__()
	
        # fine-tune
        self.subnet = ResNet_LSTM_extract(39, 2, 128, 3, 7, pretrained=True)
        # self.proj_v = nn.Linear(35, 128)  # if the input is img sequences, please change FC to Resnet
        self.proj_v1 = nn.Linear(512, 128) 
        # self.proj_a = nn.Linear(74, 128)  # B x T x 128
        # self.proj_l = nn.Linear(300, 128) # B x T x 128
        

        self.dropout = nn.Dropout(0.5)
        self.specific_projection_v = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32)
        )


        self.specific_projection_a = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32)
        )

        self.specific_projection_l = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32)
        )

        self.invariant_projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32)
        )

        self.distriminator = nn.Sequential(
            nn.Linear(32, 3)
        )


        self.transformer = CrossTransformer(source_num_frames = 72,
                                                tgt_num_frames = 72,
                                                dim=32,
                                                depth=transformer_depth,
                                                heads=8,
                                                mlp_dim=32,
                                                dropout=0.5,
                                                emb_dropout=0.1
                                                )
                                                
	# try to use non-linear classification if the performance is not well
        self.cls_head = nn.Sequential(
            # nn.BatchNorm1d(64),
            # nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(),
            # nn.Linear(16, 8),
            # nn.LeakyReLU(),
            # nn.Linear(8 , num_classs)
            # nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(),
            # nn.Linear(16, 8),
            # nn.LeakyReLU(),
            nn.Linear(32, num_classs)
        )


    def forward(self, x_vision, x_audio, x_text):
    	
    	# feature extraction
        # x_vision = self.proj_v(x_vision)
        # x_audio = self.proj_a(x_audio)
        # x_text = self.proj_l(x_text)
        
        x_vision, x_audio, x_text = self.subnet(x_vision, x_audio, x_text)
        x_vision = self.proj_v1(x_vision)

        x_vision = self.dropout(x_vision)
        x_audio = self.dropout(x_audio)
        x_text = self.dropout(x_text)
	
	# representation learning
        vision_specific = self.specific_projection_v(x_vision)
        audio_specific = self.specific_projection_a(x_audio)
        text_specific = self.specific_projection_l(x_text)
        vision_invariant = self.invariant_projection(x_vision)
        audio_invariant = self.invariant_projection(x_audio)
        text_invariant = self.invariant_projection(x_text)
	
	# GRL
        # vision_invariant = GradReverse.apply(vision_invariant, 1.0)
        # audio_invariant = GradReverse.apply(audio_invariant, 1.0)
        # text_invariant = GradReverse.apply(text_invariant, 1.0)
        
        vision_invariant_tmp = vision_invariant.clone()
        audio_invariant_tmp = audio_invariant.clone()
        text_invariant_tmp = text_invariant.clone()

        vision_invariant_cls_out = self.distriminator(GradReverse.apply(vision_invariant_tmp.clone(), 1.0).mean(dim=1))
        audio_invariant_cls_out = self.distriminator(GradReverse.apply(audio_invariant_tmp.clone(), 1.0).mean(dim=1))
        text_invariant_cls_out = self.distriminator(GradReverse.apply(text_invariant_tmp.clone(), 1.0).mean(dim=1))

        vision_specific_cls_out = self.distriminator(vision_specific.clone().mean(dim=1))
        audio_specific_cls_out = self.distriminator(audio_specific.clone().mean(dim=1))
        text_specific_cls_out = self.distriminator(text_specific.clone().mean(dim=1))
        # g1 = autograd.grad(self.distriminator.parameters(), retain_graph=True)[0]
        # print(g1)
	
	# feature fusion
        modality_invariant = torch.cat((vision_invariant, audio_invariant, text_invariant), dim=1)
        modality_specific = torch.cat((vision_specific, audio_specific, text_specific), dim=1)
        feat = self.transformer(modality_specific, modality_invariant).mean(dim=1)
	
	# classification
        emotion_cls_output = self.cls_head(feat)
        # g2 = autograd.grad(self.cls_head.parameters(), retain_graph=True)[0]
        # print(g2)

        return vision_invariant_cls_out, audio_invariant_cls_out,text_invariant_cls_out, vision_specific_cls_out, audio_specific_cls_out, text_specific_cls_out, emotion_cls_output




