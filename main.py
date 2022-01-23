import os, sys, shutil
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from model import RNN
from dataset import get_dataloader
from tqdm import trange
from utilities.common_utils import get_logger
import logging
import matplotlib.pyplot as plt
import argparse


logger = get_logger(name=__name__, log_file=None, log_level=logging.DEBUG, log_level_name='')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    #creating tensorboard object
    tb_writer = SummaryWriter(log_dir=os.path.join(args.outdir, "tb/"), purge_step=0)    

    train_dl, val_dl, vocab, label_map, train_num, val_num = get_dataloader(args.datapath)    

    criterion = nn.CrossEntropyLoss()

    vocab_size = len(vocab)
    num_classes = len(label_map)
    model = RNN(vocab_size, num_classes, args.embed_dim, args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)    
    best_accuracy = 0        
    for epoch in trange(args.epoches):
        train_loss, test_loss = 0.0, 0.0
        correct_num = 0
        
        model.train()
        for i, batch in enumerate(train_dl):
            seqs, labels = batch
            optimizer.zero_grad()
            seqs = seqs.to(device)
            pred_outputs = model(seqs)
            loss = criterion(pred_outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct_num += (torch.argmax(pred_outputs, dim=1) == labels).sum().item()

        accuracy = correct_num / train_num
        tb_writer.add_scalar('Train_Loss', train_loss, epoch)
        tb_writer.add_scalar('Train_Accuracy', accuracy, epoch)

        model.eval()
        correct_val_num = 0
        for eval_batch in val_dl:
            seqs, labels = eval_batch
            seqs = seqs.to(device)
            pred_outputs = model(seqs)
            loss = criterion(pred_outputs, labels)
            test_loss += loss.item()
            correct_val_num += (torch.argmax(pred_outputs, dim=1) == labels).sum().item()
        val_accuracy = correct_val_num / val_num
        tb_writer.add_scalar('Test_Loss', test_loss, epoch)
        tb_writer.add_scalar('Test_Accuracy', val_accuracy, epoch)        
        logger.info(
            f"Epoch : {str(epoch).zfill(2)}, "
                f"Training Loss : {round(train_loss, 4)}, Training Accuracy : {round(accuracy, 4)}, "
                f"Test Loss : {round(test_loss, 4)}, Test Accuracy : {round(val_accuracy, 4)}")

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), args.outdir + args.modelname + str(epoch))
    
    # Plot confusion matrix
    y_true = []
    y_pred = []
    for data in val_dl:
        seq, labels = data
        outputs = model(seq)
        predicted = torch.argmax(outputs, dim=1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()

    cm = confusion_matrix(np.array(y_true), np.array(y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map.keys())
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    plt.show()    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--seed", type=int, default=3, help="")
    parser.add_argument("--datapath", type=str, default="/home/qcdong/corpus_general/nlp_corpus/names_crnn/names")
    parser.add_argument("--outdir", type=str, default="./output/", help="")
    parser.add_argument("--modelname", type=str, default="model", help="")
    parser.add_argument("--epoches", type=int, default=20, help="")
    parser.add_argument("--lr", type=int, default=.005, help="")
    parser.add_argument("--embed_dim", type=int, default=128, help="")
    parser.add_argument("--hidden_size", type=int, default=128, help="")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)