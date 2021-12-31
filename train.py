from sklearn.utils import class_weight
import torch
import torch.utils.data as tud
from data_preprocess.datasets import ta_dataset, build_vocab
from models.TextCNN import TextCNN
from utils import Logger
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

if __name__ == '__main__':
    BATCH_SIZE = 64
    MAX_LEN = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR = 1e-2
    CLASS_NUM = 8
    N_EPOCHS = 50

    train_csv = pd.read_csv('raw_data/clean-ta-train.csv')
    dev_csv = pd.read_csv('raw_data/clean-ta-dev.csv')
    ta_vocab = build_vocab(train_csv, dev_csv)

    # define Dataset and Dataloader
    train_dataset = ta_dataset(train_csv, ta_vocab, max_len=MAX_LEN)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.labels), y=train_dataset.labels)
    class_weights = torch.Tensor(class_weights).to(DEVICE)

    dev_dataset = ta_dataset(dev_csv, ta_vocab, max_len=MAX_LEN)

    train_loader = tud.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = tud.DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    # define Model
    model = TextCNN(vocab_size=len(ta_vocab), embed_dim=64, class_num=CLASS_NUM, kernel_num=32, kernel_sizes=[3, 4, 5], dropout=0.5).to(DEVICE)
    
    # define Crition and Optimizer
    loss_func = torch.nn.CrossEntropyLoss(class_weights)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # define logger
    logger = Logger(N_EPOCHS, len(train_loader))
    max_test_score = 0.0

    # train
    for epoch in range(N_EPOCHS):

        test_acc = 0.0
        test_num = len(dev_loader)

        model.train()
        for i, (tokens, labels) in enumerate(train_loader):
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)

            preds = model(tokens)
            loss = loss_func(preds, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            acc = accuracy_score(labels.cpu(), torch.argmax(preds, dim=1).cpu())
            logger.log({'loss': loss, 'acc': acc})
        
        model.eval()
        with torch.no_grad():

            for i, (tokens, labels) in enumerate(dev_loader):
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)

                preds = model(tokens)
                acc = accuracy_score(labels.cpu(), torch.argmax(preds, dim=1).cpu())
                test_acc += acc
            
            test_acc /= test_num
            print('dev_acc %.4f' % test_acc)
            
            if test_acc.item() >= max_test_score:
                print("New model saved!")
                max_test_score = test_acc.item()
                torch.save(model.state_dict(), 'state_dicts/Model.pth')
