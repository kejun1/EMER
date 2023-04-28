import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from dataset.in_SMEAD_dataset import My_FEV_Dataset
from dataset.in_SMEAD_dataset import Load_FEV_Dataset
import random
import numpy as np
from model.ResNet_LSTM import ResNet_LSTM
from model.in_ResNet_Transformer import ResNet_Transformer
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn import metrics
from model.MulT import MULT
from model.TFN import TFN
from model.AEQL.in_model import AEQL
from model.AEQL.in_loss import MultimodalLoss as loss_f

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(0)

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_acc = 0.0
max_acc2 = 1.0

parser = argparse.ArgumentParser(description='SMEAD Training')
parser.add_argument('--out_path', default='./checkpoints/in_FEV_aeql/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--load_path', default='./checkpoints/in_FEV_aeql/resnet_new.pth',
                    help="path to net (to continue training)")
parser.add_argument('--log_path', default='./logs/',
                    help='folder to output logs')
parser.add_argument('--experiment_name', default='in_FEV_aeql_set1',
                    help='experiment_name')
parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=200, type=int)
parser.add_argument('--lr', dest='lr',
                        help='learning rate',
                        default=1e-4, type=float)
parser.add_argument('--bs', dest='batch_size',
                    help='batch_size',
                    default=16, type=int)
parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=5e-4, type=float)
parser.add_argument('--k_test', dest='k_test',
                        help='test set number',
                        default=1, type=int)

args = parser.parse_args()

def train(model, trainloader, criterion, optimizer, epoch, logger):
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    count = 0
    sum_acc = 0.0
    sum_f1 = 0.0
    sum_war = 0.0
    sum_uar = 0.0
    
    sum_mae = 0.0
    sum_mse = 0.0
    sum_r2 = 0.0
    sum_rmse = 0.0
    
    sum_in = 0.0
    sum_s = 0.0
    sum_cls = 0.0
    
    model.train()
    for iter, data in enumerate(trainloader):
        count += 1
        f_inputs, e_inputs, v_inputs, labels, lables_in, _,_ = data
        f_inputs, e_inputs, v_inputs, labels = f_inputs.to(device), e_inputs.to(device), v_inputs.to(device), labels.to(device)
        lables_in = lables_in.to(device)
        lables_in = torch.unsqueeze(lables_in,axis=1)

        vision_invariant_cls_out, audio_invariant_cls_out,text_invariant_cls_out, \
    vision_specific_cls_out, audio_specific_cls_out, text_specific_cls_out, emotion_cls_output, out_in = model(f_inputs, e_inputs, v_inputs)

        out_in = out_in.to(torch.float64)
        loss, modality_invariant_loss, modality_specific_loss, ClsLoss = criterion(vision_invariant_cls_out, audio_invariant_cls_out, text_invariant_cls_out, vision_specific_cls_out, 
                    audio_specific_cls_out, text_specific_cls_out, emotion_cls_output, out_in, labels, lables_in)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        sum_loss += loss.item()
        sum_in += modality_invariant_loss.item()
        sum_s += modality_specific_loss.item()
        sum_cls += ClsLoss.item()

        
        mae_k = metrics.mean_absolute_error(lables_in.cpu().numpy(), out_in.detach().cpu().numpy())
        mse_k = metrics.mean_squared_error(lables_in.cpu().numpy(),out_in.detach().cpu().numpy())
        r2_k = metrics.r2_score(lables_in.cpu().numpy(), out_in.detach().cpu().numpy())
        rmse_k = np.sqrt(mse_k)

        sum_mae += mae_k
        sum_mse += mse_k
        sum_r2 += r2_k
        sum_rmse += rmse_k
        


        _, predicted = torch.max(emotion_cls_output.data, 1)

        acc_k = metrics.accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        f1_k = metrics.f1_score(labels.cpu().numpy(),predicted.cpu().numpy(),  average='weighted')
        war_k = metrics.recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted',zero_division=0)
        uar_k = metrics.recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro',zero_division=0)

        sum_acc += acc_k
        sum_f1 += f1_k
        sum_war += war_k
        sum_uar += uar_k
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f | F1: %.3f | WAR: %.3f | UAR: %.3f'
              % (epoch, iter, sum_loss / (iter + 1), acc_k, f1_k, war_k, uar_k))
        print(' in_MAE: %.3f | in_MSE: %.3f | in_R2: %.3f | in_RMSE: %.3f '
              % ( mae_k, mse_k, r2_k, rmse_k))
        print('modality_invariant_loss: %.03f |modality_specific_loss: %.03f |ClsLoss: %.03f '
              % (sum_in / (iter + 1),sum_s / (iter + 1),sum_cls / (iter + 1)))
    
    logger.add_scalar("train_loss", sum_loss / count, global_step=epoch)
    logger.add_scalar("train_acc", sum_acc / count, global_step=epoch)
    logger.add_scalar("train_f1", sum_f1 / count, global_step=epoch)
    logger.add_scalar("train_war", sum_war / count, global_step=epoch)
    logger.add_scalar("train_uar", sum_uar / count, global_step=epoch)
    
    logger.add_scalar("train_mae_a", sum_mae / count, global_step=epoch)
    logger.add_scalar("train_mse_a", sum_mse / count, global_step=epoch)
    logger.add_scalar("train_r2_a", sum_r2 / count, global_step=epoch)
    logger.add_scalar("train_rmse_a", sum_rmse / count, global_step=epoch)
    
    logger.add_scalar("modality_invariant_loss", sum_in / count, global_step=epoch)
    logger.add_scalar("modality_specific_loss", sum_s / count, global_step=epoch)
    logger.add_scalar("ClsLoss", sum_cls / count, global_step=epoch)

    
    

def test(model, testloader, epoch, k_test, logger):
    global max_acc
    global max_acc2
    with torch.no_grad():

        sum_loss = 0.0

        count = 0
        sum_acc = 0.0
        sum_f1 = 0.0
        sum_war = 0.0
        sum_uar = 0.0
        
        sum_mae = 0.0
        sum_mse = 0.0
        sum_r2 = 0.0
        sum_rmse = 0.0
        

        model.eval()
        for iter, data in enumerate(testloader):
            count += 1
            f_inputs, e_inputs, v_inputs, labels, lables_in, _,_ = data
            f_inputs, e_inputs, v_inputs, labels = f_inputs.to(device), e_inputs.to(device), v_inputs.to(device), labels.to(device)
            lables_in = lables_in.to(device)
            

            vision_invariant_cls_out, audio_invariant_cls_out,text_invariant_cls_out, \
            vision_specific_cls_out, audio_specific_cls_out, text_specific_cls_out, emotion_cls_output, out_in = model(f_inputs, e_inputs, v_inputs)

            
            out_in = out_in.to(torch.float64)
            mae_k = metrics.mean_absolute_error(lables_in.cpu().numpy(), out_in.detach().cpu().numpy())
            mse_k = metrics.mean_squared_error(lables_in.cpu().numpy(),out_in.detach().cpu().numpy())
            r2_k = metrics.r2_score(lables_in.cpu().numpy(), out_in.detach().cpu().numpy())
            rmse_k = np.sqrt(mse_k)

            sum_mae += mae_k
            sum_mse += mse_k
            sum_r2 += r2_k
            sum_rmse += rmse_k
            

            _, predicted = torch.max(emotion_cls_output.data, 1)
            acc_k = metrics.accuracy_score(predicted.cpu().numpy(), labels.cpu().numpy())
            f1_k = metrics.f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
            war_k = metrics.recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted',zero_division=0)
            uar_k = metrics.recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro',zero_division=0)

            sum_acc += acc_k
            sum_f1 += f1_k
            sum_war += war_k
            sum_uar += uar_k

        acc = sum_acc / count
        acc2 = sum_mae / count
        if max_acc2 > acc2:
            max_acc2 = acc2
        if max_acc < acc:
            max_acc = acc
        print('loss: %.3f, set%d, test_acc：%.3f, test_f1：%.3f, test_war：%.3f, test_uar：%.3f' % (sum_loss / count, k_test, acc, sum_f1/count,sum_war/count,sum_uar/count))
        
        print('test_mae_in：%.3f, test_mse_in：%.3f, test_r2_in：%.3f, test_rmse_in：%.3f' 
              % (acc2, sum_mse/count,sum_r2/count,sum_rmse/count))
        print('set%d, max_acc: %.3f' % (k_test, max_acc))
        print('set%d, min_mae: %.3f' % (k_test, max_acc2))
        s_path = os.path.join(args.out_path,"set"+str(args.k_test))
        if not os.path.exists(s_path):
            os.makedirs(s_path)
        torch.save(model.state_dict(), '%s/%s_%03d.pth' % (s_path, args.experiment_name, epoch))
        print('Save model:%s/%s_%03d.pth' % (args.out_path, args.experiment_name, epoch))
        logger.add_scalar("test_loss", sum_loss / count, global_step=epoch)
        logger.add_scalar("test_acc", sum_acc / count, global_step=epoch)
        logger.add_scalar("test_f1", sum_f1 / count, global_step=epoch)
        logger.add_scalar("test_war", sum_war / count, global_step=epoch)
        logger.add_scalar("test_uar", sum_uar / count, global_step=epoch)
        logger.add_scalar("test_mae_v", sum_mae / count, global_step=epoch)
        logger.add_scalar("test_mse_v", sum_mse / count, global_step=epoch)
        logger.add_scalar("test_r2_v", sum_r2 / count, global_step=epoch)
        logger.add_scalar("test_rmse_v", sum_rmse / count, global_step=epoch)


def main():
    args = parser.parse_args()
    print(args)
    trainloader, testloader = Load_FEV_Dataset(v_train_path=r"/data2/lkj/SMEAD/data/smead_cross/face",
                                             v_test_path=r"/data2/lkj/SMEAD/data/smead_cross/face",
                                             e_train_path=r"/data2/lkj/SMEAD/data/smead_cross/eye",
                                             e_test_path=r"/data2/lkj/SMEAD/data/smead_cross/eye",
                                               k_test=args.k_test,
                                             batch_size=args.batch_size)
    logger = SummaryWriter(log_dir=os.path.join(args.log_path, args.experiment_name))

    model = AEQL(7,2).to(device)


    criterion = loss_f(0.3, 0.1, 1.0, batch_size=args.batch_size, device='cuda') 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, criterion,  optimizer, epoch, logger)
        lr_schduler.step()
        test(model, testloader, epoch, args.k_test, logger)

    logger.close()

if __name__ == '__main__':
    main()
