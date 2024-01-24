import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from dataset.in_SMEAD_dataset import My_FEV_Dataset
from dataset.in_SMEAD_dataset import Load_FEV_Dataset
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn import metrics
from model.MAQT.in_model import MAQT


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='SMEAD Intensity Task Test')
parser.add_argument('--out_path', default='./checkpoints/in_FEV_aeql/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--load_path', default='./checkpoints/Intensity_model/in_MATmer_set5.pth',
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

total_mae = 0.0
total_mse = 0.0
total_rmse = 0.0

def test(model, testloader, set_k):
    global total_mae
    global total_mse
    global total_rmse
    with torch.no_grad():
        count = 0
        sum_mae = 0.0
        sum_mse = 0.0
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
            rmse_k = np.sqrt(mse_k)

            sum_mae += mae_k
            sum_mse += mse_k
            sum_rmse += rmse_k

        acc2 = sum_mae / count
        total_mae += sum_mae / count
        total_mse += sum_mse / count
        total_rmse += sum_rmse / count

        print('set: %d, test_mae_in：%.3f, test_mse_in：%.3f, test_rmse_in：%.3f'
              % (set_k, acc2, sum_mse/count, sum_rmse/count))



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

        load_path = "./checkpoints/Intensity_model/in_MATmer_set" + str(k_set) + ".pth"

        model = MAQT(7, 2).to(device)
        model.load_state_dict(torch.load(load_path))
        print("Loaded model!")


        test(model, testloader, k_set)

    print('Total: test_mae_in：%.3f, test_mse_in：%.3f, test_rmse_in：%.3f'
              % (total_mae/5, total_mse/5, total_rmse/5))

if __name__ == '__main__':
    main()
