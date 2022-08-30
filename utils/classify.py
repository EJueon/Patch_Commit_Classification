import matplotlib.pyplot as plt
import torch
import preprocess as pre
from model import LSTM
from dataset import CodeDataset
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


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
    total = len(y_pred)
    accuracy = count/total
    print(f'{count}/{total}')
    print(f'accuracy : {accuracy}')
    # print('Classification Report:')
    # print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    # cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    # ax= plt.subplot()
    # sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    # ax.set_title('Confusion Matrix')

    # ax.set_xlabel('Predicted Labels')
    # ax.set_ylabel('True Labels')

    # ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    # ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
    # plt.show() 
    return count, total, accuracy

def load(model_path, vocab_size=32):
    model = LSTM(vocab_size=vocab_size, embedding_dim=128, hidden_size=128).to(device)
    if model_path:
        print('Loading the model from', model_path)
        model.load_state_dict(torch.load(model_path))
    return model

def show_accuracy(model, data_path, tokenizer, batch_size=32):
    vocab_size = 10000
    train, valid, test = pre.load_data(data_path)
    test_dataset = CodeDataset(test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    count, total, acc = evaluate(model, test_loader, tokenizer)
    return count, total, acc

# def classify(model, data, tokenizer, batch_size=1):
    
    

# if __name__ == '__main__':
#     filepath='syzbot_data.pickle'
#     tokenizer_path = './vulpatch-tokenizer'
#     batch_size = 32
#     vocab_size = 10000
    
#     train, valid, test = pre.load_data(filepath)
#     tokenizer = pre.load_tokenizer(tokenizer_path)
#     # tokenizer = pre.vocab_hugginface(train, vocab_size)
#     test_dataset = CodeDataset(test, tokenizer)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
#     # model = load('./model.pt')
#     # evaluate(model, test_loader, tokenizer)
