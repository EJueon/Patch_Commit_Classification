# https://colab.research.google.com/drive/1cpn6pk2J4liha9jgDLNWhEWeWJb2cdch?usp=sharing#scrollTo=CYI5TflH4-jG
import os
from dataset import CodeDataset
import preprocess as pre
import torch
import torch.nn as nn
from model import LSTM
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def load(model_path, model):
    """ load saved model or pretrained transformer (a part of model) """
    if model_path:
        print('Loading the model from', model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))

    
def save(save_path, model):
    """ save current model """
    torch.save(model.state_dict(), # save model object before nn.DataParallel
                os.path.join(save_path, 'model.pt'))
    print(f'Model saved to ==> {save_path}')

def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, optimizer):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Evaluation Function
def evaluate(model, test_loader, tokenizer, batch_size = 32, version='title', threshold=0.5):
    # Evaluation

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for name, input_batch, labels in test_loader:           
            labels = labels.to(device)
            input_batch = input_batch.to(device)
            output = model(input_batch, batch_size)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
    
    count = 0
    for seq, pred, true in zip(input_batch, y_pred, y_true):
        if pred == true:
            print(tokenizer.decode(seq.detach().cpu().numpy()[0]))
            count += 1
    print(f'{count}/{len(y_pred)}')
    print(f'accuracy : {count/len(y_pred)}')
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
    plt.show() 

def train_epoch(model,
          optimizer,
          scheduler,
          train_loader,
          valid_loader,
          batch_size, 
          criterion = nn.BCELoss(),
          num_epochs = 5,
          file_path = './',
          best_valid_loss = float("Inf")):
    
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    eval_every = batch_size
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for _, input_batch, labels in train_loader:           
            labels = labels.to(device)
            input_batch = input_batch.to(device)
            output = model(input_batch, batch_size)

            loss = criterion(output.float(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                  for _, input_batch, labels in valid_loader:
                      labels = labels.to(device)
                      input_batch = input_batch.to(device)
                      output = model(input_batch, batch_size)

                      loss = criterion(output.float(), labels.float())
                      valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                scheduler.step(average_valid_loss)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
    
                 # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save(file_path, model)
                    # save_checkpoint(os.path.join(file_path + 'model.pt'), model, optimizer, best_valid_loss)
                    # save_metrics(os.path.join(file_path + 'metrics.pt'), train_loss_list, valid_loss_list, global_steps_list)
    # save_metrics(os.path.join(file_path, 'metrics.pt'), train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')



def main():
    
    batch_size = 32
    
    # train, valid, test = pre.load_data(filepath)

    train = pre.load_data('./data/train_data.pickle.gz')
    valid = pre.load_data('./data/valid_data.pickle.gz')
    test = pre.load_data('./data/test_data.pickle.gz')

    tokenizer = pre.load_tokenizer('./vulpatch-tokenizer')
    vocab_size = tokenizer.vocab_size
    maxlen = 512

    train_dataset = CodeDataset(train, tokenizer, max_len=maxlen)
    valid_dataset = CodeDataset(valid, tokenizer, max_len=maxlen)
    test_dataset = CodeDataset(test, tokenizer, max_len=maxlen)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = LSTM(vocab_size=vocab_size, embedding_dim=128, hidden_size=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    train_epoch(model=model, 
          optimizer=optimizer, 
          scheduler=scheduler,
          train_loader=train_loader,
          valid_loader=valid_loader,
          batch_size = batch_size, 
          num_epochs=50,
          file_path='./data')

    
    evaluate(model, test_loader, tokenizer)
    
if __name__ == '__main__':
    main()