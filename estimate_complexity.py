import os
import sys
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from torch.nn.parallel import DataParallel

sys.path.insert(0,"./influence_function/")
from influence_utils import EKFACInfluence, EKFAC
from torchvision import datasets, transforms


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument("--reference_model_num",default=5,type=int,help="number of reference models",)
    parser.add_argument("--dataset",default="cifar10",type=str,help="dataset",)
    args = parser.parse_args()


    set_random_seed(2024)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


    
    if args.dataset=='cifar10':
        # Load Reference Model
        from cifar10 import convnet as CIFAR_net
        model = CIFAR_net(num_classes=10)#.to(device)

        # Load Dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root="/home/lip/LiPan/data/CIFAR10/", train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root="/home/lip/LiPan/data/CIFAR10/", train=False, download=True, transform=transform)
        
    elif args.dataset=='mnist':
        from mnist import SampleConvNet
        model = SampleConvNet()#.to(device)

        train_dataset = datasets.MNIST(
            "/home/lip/LiPan/data/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        test_dataset = datasets.MNIST(
            "/home/lip/LiPan/data/",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
    elif args.dataset=='fmnist':
        from fmnist import SampleConvNet
        model = SampleConvNet()#.to(device)

        train_dataset = datasets.FashionMNIST("/home/lip/LiPan/data", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        test_dataset = datasets.FashionMNIST("/home/lip/LiPan/data", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    elif args.dataset=='imdb':
        from imdb import SampleNet
        from datasets import load_dataset
        from transformers import BertTokenizerFast

        raw_dataset = load_dataset("imdb", cache_dir="/home/lip/LiPan/data/IMDB")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        dataset = raw_dataset.map(
            lambda x: tokenizer(
                x["text"], truncation=True, max_length=args.max_sequence_length
            ),
            batched=True,
        )
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        model = SampleNet(vocab_size=len(tokenizer))#.to(device)


    #model = DataParallel(model).to(device)
    model = model.to(device)

    #########################################################################################
    src_dataset = Subset(train_dataset, range(100)) # train_dataset
    query_dataset = Subset(train_dataset, range(100)) #train_dataset 


    # Compute complexity
    complexity_matrix = None
    for i in range(1, args.reference_model_num+1):
        if not os.path.exists('{}_reference_model_{}.tar'.format(args.dataset, i)):
            print("Wrong! This reference model dosen't exist!")
            continue

        checkpoint = torch.load('{}_reference_model_{}.tar'.format(args.dataset, i))
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        try:
            model.load_state_dict(checkpoint)
        except:
            new_checkpoint = {}
            for key in list(checkpoint.keys()):
                new_checkpoint['module.'+key] = checkpoint[key]
            model.load_state_dict(new_checkpoint)
            model = model.module


        '''
        # Return all learnable parameters
        for name,param in model.named_parameters():
            print(name)
            # [0.weight,0.bias,3.weight,3.bias,6.weight,6.bias,9.weight,9.bias,13.weight,13.bias]
        
        # Return all children modules in surface
        for name,param in model.named_children():
            print(name)
            # [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

        # Return all children modules in deep 
        for name,param in model.named_modules():
            print(name)
            # [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        '''

        if args.dataset=='cifar10':
            layers=['13'] # only consider fully-connected layer
            '''
            layers = []
            for name,param in model.named_parameters():
                if name.split('.')[0] not in layers:
                    layers.append(name.split('.')[0])
            '''
        elif args.dataset=='mnist':
            layers=['fc2']
        elif args.dataset=='fmnist':
            layers=['fc2']
        elif args.dataset=='imdb':
            layers=['fc2']



        influence_model = EKFACInfluence(model, layers=layers, influence_src_dataset=src_dataset, batch_size=1, cov_batch_size=1)
        influence_model.influence(query_dataset)
        sys.exit()

        complexity = influence_model.complexity(query_dataset, eps=1e-4)

        if i==1:
            complexity_matrix = complexity.numpy().reshape(-1,1)
        else:
            complexity_matrix = np.concatenate((complexity_matrix, complexity.numpy().reshape(-1,1)), axis=1)
    
    np.save(args.dataset+"_complexity_matrix.npy", complexity_matrix)
    # complexity_matrix = np.load(args.dataset+"_complexity_matrix.npy")
    




if __name__ == '__main__':
    main()
