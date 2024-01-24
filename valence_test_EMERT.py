import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from dataset.SMEAD_dataset import My_FEV_Dataset
from dataset.SMEAD_dataset import Load_FEV_Dataset
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn import metrics
from model.MAQT.model import MAQT





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='SMEAD Valence Task Test')
parser.add_argument('--out_path', default='./checkpoints/valence_test_MATmer/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--load_path', default='./checkpoints/valence_test_MATmer/resnet_new.pth',
                    help="path to net (to continue training)")
parser.add_argument('--log_path', default='./logs/',
                    help='folder to output logs')
parser.add_argument('--experiment_name', default='valence_test_MATmer_set1',
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
parser.add_argument('--seed', dest='seed',
                        help='test seed number',
                        default=0, type=int)

args = parser.parse_args()

total_mae = 0.0
total_mse = 0.0
total_rmse = 0.0

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

def test(model, testloader, k_test):
    global total_mae
    global total_mse
    global total_rmse

    with torch.no_grad():

        sum_loss = 0.0
        count = 0
        sum_mae = 0.0
        sum_mse = 0.0
        sum_r2 = 0.0
        sum_rmse = 0.0

        model.eval()
        for iter, data in enumerate(testloader):
            count += 1
            f_inputs, e_inputs, v_inputs, labels, lv,la= data
            f_inputs, e_inputs, v_inputs, labels = f_inputs.to(device), e_inputs.to(device), v_inputs.to(
                device), labels.to(device)
            lv,la = lv.to(device),la.to(device)
            lv = torch.unsqueeze(lv, axis=1)
            

            vision_invariant_cls_out, audio_invariant_cls_out,text_invariant_cls_out, \
    vision_specific_cls_out, audio_specific_cls_out, text_specific_cls_out, emotion_cls_output = model(f_inputs, e_inputs, v_inputs)

            out_v = emotion_cls_output.to(torch.float64)
            
            mae_k = metrics.mean_absolute_error(lv.cpu().numpy(), out_v.detach().cpu().numpy())
            mse_k = metrics.mean_squared_error(lv.cpu().numpy(),out_v.detach().cpu().numpy())
            rmse_k = np.sqrt(mse_k)

            sum_mae += mae_k
            sum_mse += mse_k
            sum_rmse += rmse_k
            

        acc = sum_mae / count
        total_mae += sum_mae / count
        total_mse += sum_mse / count
        total_rmse += sum_rmse / count

        print('set%d, test_mae_v：%.3f, test_mse_v：%.3f, test_rmse_v：%.3f'
              % (k_test, acc, sum_mse/count, sum_rmse/count))





def main():
    args = parser.parse_args()
    print(args)
    for k_set in range(1, 6):
        trainloader, testloader = Load_FEV_Dataset(v_train_path=r"./data/smead_cross/face",
                                                   v_test_path=r"./data/smead_cross/face",
                                                   e_train_path=r"./data/smead_cross/eye",
                                                   e_test_path=r"./data/smead_cross/eye",
                                                   k_test=k_set,
                                                   batch_size=args.batch_size)

        load_path = "./checkpoints/Valence_model/valence_MATmer_set" + str(k_set) + ".pth"

        model = MAQT(1, 2).to(device)
        model.load_state_dict(torch.load(load_path))
        print("Loaded model!")

        test(model, testloader, k_set)

    print('Total: test_mae_v：%.3f, test_mse_v：%.3f, test_rmse_v：%.3f'
          % (total_mae / 5, total_mse / 5, total_rmse / 5))

if __name__ == '__main__':
    main()
