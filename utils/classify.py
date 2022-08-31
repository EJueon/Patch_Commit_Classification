import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataset import CodeDataset
from utils.model import LSTM
import utils.preprocess as pre
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Evaluation Function
def evaluate(model, test_loader, tokenizer, batch_size = 1, version='title', threshold=0.5):
    
    y_pred = []
    y_true = []
    names = []
    model.eval()
    with torch.no_grad():
        for name, input_batch, labels in test_loader:           
            labels = labels.to(device)
            input_batch = input_batch.to(device)
            output = model(input_batch, batch_size)

            output = (output > threshold).int()
            names.append(name)
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
    
    count = 0

    vuls = []
    for name, pred, true in zip(names, y_pred, y_true):
        if pred == 1:
            vuls.append(name[0])
        if pred == true:
            # print(tokenizer.decode(seq.detach().cpu().numpy()[0]))
            count += 1
    total = len(y_pred)
    accuracy = count/total
    print(f'{count}/{total}')
    print(f'accuracy : {accuracy*100:.2f}')

    return count, total, accuracy, vuls

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

def _load(model_path, model):
    """ load saved model or pretrained transformer (a part of model) """
    if model_path:
        print('Loading the model from', model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))

def _save(save_path, model):
    """ save current model """
    torch.save(model.state_dict(), # save model object before nn.DataParallel
                os.path.join(save_path, 'model.pt'))

def calc_accuracy(model, data_path, tokenizer, batch_size=1):
    test = pre.load_data(data_path)
    test_dataset = CodeDataset(test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    count, total, acc, vuls = evaluate(model, test_loader, tokenizer)
    return count, total, acc, vuls



if __name__ == '__main__':
    
    tokenizer = pre.load_tokenizer('./vulpatch-tokenizer')
    vocab_size = tokenizer.vocab_size
    model_path = './data/model.pt'
    model = LSTM(vocab_size=vocab_size, embedding_dim=128, hidden_size=128).to(device)
    try:
        _load(model_path, model)
    except:
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        load_checkpoint(model_path, model, optimizer)
    calc_accuracy(model, './data/test_data.pickle.gz', tokenizer)
