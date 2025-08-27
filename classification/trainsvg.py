import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import wandb  # delete when commit
from svg_dataset import SvgDataset
from available_models import load_model
import argparse

"""
This training is on svg dataset 
"""


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main(args):
    wandb.init(
        project="svg_classification",
        name=args.model_name + '_pretrained' + '_' + str(args.num_shapes) + '_shapes_mode_' + str(
            args.mode) + '_sca_' + args.data_scal_factor if args.use_pretrain else args.model_name + '_' + str(
            args.num_shapes) + '_shapes_mode_' + str(args.mode) + '_sca_' + args.data_scal_factor,
        config={
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'model': args.model_name,
            'dataset': args.dataset
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    dataset_path = os.path.join(args.data_path, args.dataset)
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomErasing(),

        ]),
        'val': transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),  # keep size as 224*224
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    train_dataset = SvgDataset(root_paths=os.path.join(dataset_path, 'train'),
                               num_shapes=args.num_shapes,
                               mode=args.mode,
                               transform=data_transform['train'],
                               scaling=args.data_scal_factor
                               )
    val_dataset = SvgDataset(root_paths=os.path.join(dataset_path, 'val'),
                             num_shapes=args.num_shapes,
                             mode=args.mode,
                             transform=data_transform['val']
                             )
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    # load the model
    model = load_model(args.model_name, pretrained=args.use_pretrain, num_classes=args.num_classes)
    if args.use_pretrain:
        logger = get_logger(
            './logfile/{}/explog_pretrained_{}epochs_on_{}%{}_{}shapes_mode{}.log'.format(args.model_name,
                                                                                          args.epochs,
                                                                                          args.data_scal_factor * 100,
                                                                                          args.dataset,
                                                                                          args.num_shapes,
                                                                                          args.mode)
        )
    else:
        logger = get_logger(
            './logfile/{}/explog_{}epochs_on_{}%{}_{}shapes_mode{}.log'.format(args.model_name,
                                                                               args.epochs,
                                                                               args.data_scal_factor * 100,
                                                                               args.dataset,
                                                                               args.num_shapes,
                                                                               args.mode)
        )
    model.to(device)

    # training, eva model, logger
    loss = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=args.lr)
    best_acc = 0
    for epoch in range(args.epochs):
        loss_sum = 0.0
        model.train()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            labels, images = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss_value = loss(logits, labels.to(device))
            loss_value.backward()
            optimizer.step()
            loss_sum += loss_value.item()
            train_bar.desc = "training epoch[{}/{}], loss: {:.5f}, lr: {}".format(epoch + 1, args.epochs,
                                                                                  loss_value,
                                                                                  optimizer.param_groups[0]['lr'])
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_labels, val_images = val_data
                val_logits = model(val_images.to(device))
                predict = torch.max(val_logits, dim=1)[1]  # torch.max()[1] returns the index of max value in tensor
                acc += torch.eq(predict, val_labels.to(device)).sum().item()  # item seems more precise
                val_bar.desc = 'validate epoch[{}/{}]'.format(epoch + 1, args.epochs)

        acc = acc / val_num
        average_loss = loss_sum / train_num
        print('epoch[{}/{}]: val_acc: {:.3f}, average_loss: {:.5f}, lr: {}'.format(epoch + 1, args.epochs, acc,
                                                                                   average_loss,
                                                                                   optimizer.param_groups[0]['lr']))
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}\t lr={:.5f} '.format(epoch + 1, args.epochs,
                                                                                   average_loss,
                                                                                   acc,
                                                                                   optimizer.param_groups[0][
                                                                                       'lr']))
        wandb.log({'epoch': epoch, 'loss': average_loss, 'accuracy': acc})
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       os.path.join(args.saved_model_path,
                                    '{}_{}epochs_on_{}_{}shapes_mode{}.pth'.format(args.model_name,
                                                                                   args.epochs,
                                                                                   args.dataset,
                                                                                   args.num_shapes,
                                                                                   args.mode)
                                    )
                       )
    wandb.log({'best_acc': best_acc})
    print("Training stage finished! best acc: {:.3f}".format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.0001)

    # dataset
    parser.add_argument('--data_path', type=str, default='../dataset')
    parser.add_argument('--dataset', type=str, default='miniImageNet_svg')
    parser.add_argument('--data_scal_factor', type=float, default=0.3)

    # set the number of shapes and mode
    parser.add_argument('--num_shapes', type=int, default=100)
    parser.add_argument('--mode', type=int, default=0)

    # set the correct pretrained model name
    parser.add_argument('--saved_model_path', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default='mv2')
    parser.add_argument('--use_pretrain', type=bool, default=False)

    opt = parser.parse_args()
    main(opt)
