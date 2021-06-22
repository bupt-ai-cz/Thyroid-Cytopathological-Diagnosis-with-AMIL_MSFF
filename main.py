import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data_utils
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import numpy as np
import random
import argparse
import os

parser = argparse.ArgumentParser(description='Multi-instance learning')
parser.add_argument('-d', '--dataset', default='thyroid', choices=['thyroid', 'breast'])
parser.add_argument('-b', '--backbone', default='resnet18', choices=['resnet18', 'resnet34'])
parser.add_argument('-m', '--method', default='BFA',choices=['B', 'BF', 'BFA'],
    help='B:baseline; BF:baseline+fpn; BFA:baseline+FPN+attention;')
parser.add_argument('-p', '--mode', default='train', choices=['train', 'test'])
parser.add_argument('-l', '--loader', default='formal', choices=['formal', 'debug'],
    help='debug mode will use the dataloader that load the data during the training instead of load the whole dataset at the begining')
parser.add_argument('-s', '--save_weight', default='./params.pkl')
parser.add_argument('-w', '--load_weight', default='./params.pkl')
parser.add_argument('-r', '--random_seed', default=4, type=int)
parser.add_argument('-g', '--gpu', default='0')
args = parser.parse_args()

model_name_dict = {'B' : 'baseline', 'BF' : 'baseline+fpn',  
                   'BFA' : 'baseline+FPN+attention'}

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  
# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(args.random_seed)

# set dataloader
if args.loader == 'formal':
    from utils.data_loader import ThyroidDataset
elif args.loader == 'debug':
    from utils.data_loader_new import ThyroidDataset

# set model
if args.method == 'B':
    from network.baseline import ResnetAttention
elif args.method == 'BF':
    from network.fpn import ResnetAttention
elif args.method == 'BFA':
    from network.fpn_attention import ResnetAttention


# set backbone
def backbone_network(num_classes=2, pretrained=True):
    if args.backbone == 'resnet18':
        model = models.resnet18(pretrained)
    elif args.backbone == 'resnet34':
        model = models.resnet34(pretrained)
    return ResnetAttention(model, num_classes)

# set dataset
if args.dataset == 'thyroid':
    img_path = '../data/old_aug/'
    train_csv_path = '../data/old_csv/multi_layer_15.csv'
    test_csv_path = '../data/old_csv/val_multi_15.csv'
elif args.dataset == 'breast':
    img_path = '../data/breast/'
    train_csv_path = '../data/breast_csv/train.csv'
    test_csv_path = '../data/breast_csv/test.csv'  

print('loading model {}, using backbone {}, random seed {}, {} dataset'.format(model_name_dict[args.method], args.backbone, args.random_seed, args.dataset) )

model = backbone_network(num_classes=2, pretrained=True)
model.cuda()
cudnn.benchmark = True


if args.mode == 'train':
    print('loading training set')
    train_loader = data_utils.DataLoader(ThyroidDataset(img_path = img_path, csv_path = train_csv_path), batch_size=1,shuffle=True)
print('loading test set')
test_loader = data_utils.DataLoader(ThyroidDataset(img_path = img_path, csv_path = test_csv_path),
                                         batch_size=1,shuffle=False)
print('finish loading')
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


def test(epoch, save_model=False):
    correct = 0
    test_loss = 0.0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    best_acc = 0

    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data,label in test_loader:
            bag_label = label[0]
            data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            output = model.forward(data) 
            if args.mode == 'test':
                real_label.append(bag_label)
                predicted_label.append(output)
            running_loss = criterion(output, bag_label)
            test_loss += running_loss.item()
            predicted = torch.max(output,1)[1] 
            if predicted == int(bag_label.item()):
                correct = correct + 1
            if predicted==0 and int(bag_label.item())==0:
                true_negative += 1
            elif predicted==0 and int(bag_label.item())==1:
                false_negative += 1
            elif predicted==1 and int(bag_label.item())==0:
                false_positive += 1
            elif predicted==1 and int(bag_label.item())==1:
                true_positive += 1        
        try:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * precision * recall / (precision + recall)
        except:
            precision = 0
            recall = 0
            f1_score = 0
    acc = correct / len(test_loader)
    print('epoch:{}, Test Loss:{:.3f}, Test Acc:{:.3f}, precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(epoch,test_loss / len(test_loader), acc, precision, recall, f1_score))

    if save_model and acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), args.save_weight)

    test_loss = 0.0
    correct = 0

def train(epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, bag_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = torch.max(output,1)[1]
        if predicted == int(bag_label.item()):
            correct = correct + 1        
        if predicted==0 and int(bag_label.item())==0:
            true_negative += 1
        elif predicted==0 and int(bag_label.item())==1:
            false_negative += 1
        elif predicted==1 and int(bag_label.item())==0:
            false_positive += 1
        elif predicted==1 and int(bag_label.item())==1:
            true_positive += 1

    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * precision * recall / (precision + recall)
    except:
        precision = 0
        recall = 0
        f1_score = 0

    print('epoch:{}, Train Loss:{:.3f} | Train Acc:{:.3f}'.format(epoch, train_loss / len(train_loader),correct / len(train_loader)))
    train_loss = 0.0
    correct = 0

if __name__ == '__main__':
    print(args)
    if args.mode == 'train':        
        for epoch in range(70):
            train(epoch)
            test(epoch, save_model=True)
            torch.cuda.empty_cache()
        torch.save(model.state_dict(), args.save_weight)
    if args.mode == 'test':
        predicted_label = []
        real_label = []
        model.load_state_dict(torch.load(args.load_weight))
        model.cuda()
        test(0)
        for i in range(len(predicted_label)):
            predicted_label[i] = predicted_label[i].squeeze()      
            predicted_label[i] = predicted_label[i].cpu().numpy()
            predicted_label[i] = np.exp(predicted_label[i])
            predicted_label[i] = predicted_label[i] / predicted_label[i].sum()
            predicted_label[i] = predicted_label[i][1]

            real_label[i] = real_label[i].cpu().numpy()
            real_label[i] = real_label[i][0]

