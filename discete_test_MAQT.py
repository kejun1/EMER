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
from model.MAQT.model_D import MAQT


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='SMEAD Classification Task Test')
parser.add_argument('--out_path', default='./checkpoints/discrete_test_AEQL/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--load_path', default='./checkpoints/discrete_test_AEQL/resnet_new.pth',
                    help="path to net (to continue training)")
parser.add_argument('--log_path', default='./logs/',
                    help='folder to output logs')
parser.add_argument('--experiment_name', default='discrete_test_AEQL_set1',
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


total_war = 0.0
total_uar = 0.0
total_f1 = 0.0



def test(model, testloader, k_test):
    global total_war
    global total_uar
    global total_f1
    with torch.no_grad():
        count = 0
        sum_f1 = 0.0
        sum_war = 0.0
        sum_uar = 0.0

        model.eval()
        for iter, data in enumerate(testloader):
            count += 1
            f_inputs, e_inputs, v_inputs, labels, _, _ = data
            f_inputs, e_inputs, v_inputs, labels = f_inputs.to(device), e_inputs.to(device), v_inputs.to(
                device), labels.to(device)
            

            vision_invariant_cls_out, audio_invariant_cls_out,text_invariant_cls_out, \
    vision_specific_cls_out, audio_specific_cls_out, text_specific_cls_out, emotion_cls_output = model(f_inputs, e_inputs, v_inputs)


            _, predicted = torch.max(emotion_cls_output.data, 1)

            f1_k = metrics.f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
            war_k = metrics.recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted',zero_division=0)
            uar_k = metrics.recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro',zero_division=0)


            sum_f1 += f1_k
            sum_war += war_k
            sum_uar += uar_k

        total_f1 += sum_f1 / count
        total_war += sum_war / count
        total_uar += sum_uar / count

        print('set%d, test_f1：%.3f, test_war：%.3f, test_uar：%.3f' % (k_test, sum_f1/count,sum_war/count,sum_uar/count))



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

        load_path = "./checkpoints/Discrete_model/discrete_MATmer_set" + str(k_set) + ".pth"

        model = MAQT(7, 2).to(device)
        model.load_state_dict(torch.load(load_path))
        print("Loaded model!")

        test(model, testloader, k_set)

    print('Total: test_war：%.3f, test_uar：%.3f, test_f1：%.3f'
          % (total_war / 5, total_uar / 5, total_f1 / 5))

if __name__ == '__main__':
    main()
