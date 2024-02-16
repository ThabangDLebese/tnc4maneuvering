import os
import torch
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tnc4maneuvering.models import WFClassifier, WFPcaEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = datetime.now()

def epoch_run(model, dataloader, train=False, lr=0.01):
    if train:
        model.train()
    else:
        model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []
    for x, y in dataloader:
        y = y.to(device)
        x = x.to(device)
        prediction = model(x)
        state_prediction = torch.argmax(prediction, dim=1)
        loss = loss_fn(prediction, y.long())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_all.append(y.cpu().detach().numpy())
        prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())

        epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
        epoch_loss += loss.item()
        batch_count += 1
    del x, y
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    epoch_auc = roc_auc_score(y_onehot_all, prediction_all, average='samples')
    epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, epoch_auprc, c


def epoch_run_encoder(encoder, classifier, dataloader, train=False, lr=0.01):
    if train:
        classifier.train()
        encoder.train()
    else:
        classifier.eval()
        encoder.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []
    for x, y in dataloader:
        y = y.to(device)
        x = x.to(device)
        encodings = encoder(x)
        prediction = classifier(encodings)
        state_prediction = torch.argmax(prediction, dim=1)
        loss = loss_fn(prediction, y.long())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_all.append(y.cpu().detach().numpy())
        prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())

        epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
        epoch_loss += loss.item()
        batch_count += 1
    del x, y
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    epoch_auc = roc_auc_score(y_onehot_all, prediction_all, average='samples')
    epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, epoch_auprc, c


def train(train_loader, valid_loader, classifier, lr, data_type, encoder=None, n_epochs=1, type='e2e', cv=0):
    best_auc, best_acc, best_aupc, best_loss = 0, 0, 0, np.inf
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    for epoch in range(n_epochs):
        if type=='e2e': 
            train_loss, train_acc, train_auc, train_auprc, _ = epoch_run(classifier, dataloader=train_loader, train=True, lr=lr)
            test_loss, test_acc, test_auc, test_auprc, _ = epoch_run(classifier, dataloader=valid_loader, train=False)
        else: 
            train_loss, train_acc, train_auc, train_auprc, _  = epoch_run_encoder(encoder=encoder, classifier=classifier, dataloader=train_loader, train=True, lr=lr)
            test_loss, test_acc, test_auc, test_auprc, _  = epoch_run_encoder(encoder=encoder, classifier=classifier, dataloader=valid_loader, train=False)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_loss<best_loss:
            best_auc = test_auc
            best_acc = test_acc
            best_loss = test_loss
            best_aupc = test_auprc
            if type == 'e2e':
                state = {
                    'epoch': epoch,
                    'state_dict': classifier.state_dict(),
                    'best_accuracy': test_acc,
                    'best_accuracy': best_auc
                }
            else:
                state = {
                    'epoch': epoch,
                    'state_dict': torch.nn.Sequential(encoder, classifier).state_dict(),
                    'best_accuracy': test_acc,
                    'best_accuracy': best_auc
                }
            if not os.path.exists( './ckpt/classifier/%s'%data_type):
                os.mkdir( './ckpt/classifier/%s'%data_type)
            torch.save(state, './ckpt/classifier/%s/%s_checkpoint_%d.pth.tar'%(data_type, type, cv))

    # Save performance plots
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses, label="train Loss")
    plt.plot(np.arange(n_epochs), test_losses, label="test Loss")
    plt.plot(np.arange(n_epochs), train_accs, label="train Acc")
    plt.plot(np.arange(n_epochs), test_accs, label="test Acc")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join("./plots/%s" % data_type, "prunnedPCA_classification_%s_%d.png"%(type, cv)), dpi=100)
    return best_acc, best_auc, best_aupc


def save_latent_representations(encoder, dataloader, output_file):
    encoder.eval()
    encodings_list, labels_list = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.numpy()
            encodings = encoder(x)  # Uses WFPcaEncoder for encoding
            encodings_list.append(encodings.cpu().numpy())
            labels_list.append(y)
    # encodings_array = np.concatenate(encodings_list, axis=0)
    # labels_array = np.concatenate(labels_list, axis=0)
    # np.savetxt(output_file, encodings_array, delimiter=",")
    # print("Reduced encodings saved to", output_file)

def run_test_pca(data, tnc_lr, data_path, window_size, n_cross_val):
    # Load data
    with open(os.path.join(data_path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(data_path, 'state_train.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(data_path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(data_path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    T = x.shape[-1]
    x_window = np.split(x[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window = np.concatenate(np.split(y[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window = torch.Tensor(np.concatenate(x_window, 0))
    y_window = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window]))

    x_window_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window_test = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window_test = torch.Tensor(np.concatenate(x_window_test, 0))
    y_window_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window_test]))

    testset = torch.utils.data.TensorDataset(x_window_test, y_window_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)
    
    del x, y, x_test, y_test
    tnc_accs, tnc_aucs, tnc_auprcs = [], [], []

    for cv in range(n_cross_val):
        shuffled_inds = list(range(len(x_window)))
        random.shuffle(shuffled_inds)
        x_window = x_window[shuffled_inds]
        y_window = y_window[shuffled_inds]
        n_train = int(0.7*len(x_window))
        X_train, X_test = x_window[:n_train], x_window[n_train:]
        y_train, y_test = y_window[:n_train], y_window[n_train:]

        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        validset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=200, shuffle=False)

        if data == f'{args.dataset}': 
            encoding_size = 64 #Orignial No components
            n_classes = 4
            num_pca_components = 6 #No of PCA components (taken from the regression task)
            tnc_encoder = WFPcaEncoder(encoding_size=encoding_size, num_pca_components=num_pca_components).to(device)

            if not os.path.exists('./ckpt/one_ds/checkpoint_%d.pth.tar' % cv):
                RuntimeError('Checkpoint for TNC4M encoder does not exist!')
            tnc_checkpoint = torch.load('./ckpt/one_ds/checkpoint_%d.pth.tar' % cv)
            tnc_encoder.load_state_dict(tnc_checkpoint['encoder_state_dict'])
            tnc_encoder.eval()

            tnc_classifier = WFClassifier(encoding_size=num_pca_components, output_size=n_classes)
            n_epochs = 10

        best_acc_tnc, best_auc_tnc, best_auprc_tnc = train(train_loader, valid_loader, tnc_classifier, tnc_lr, encoder=tnc_encoder, data_type=data, n_epochs=n_epochs, type='TNC4Maneuvering', cv=cv)
        print('TNC4Maneuvering: ', best_acc_tnc * 100, best_auc_tnc, best_auprc_tnc)
        
        # Evaluate TNC model on the test set
        if data == f'{args.dataset}':
            _, test_acc_tnc, test_auc_tnc, test_auprc_tnc, _ = epoch_run_encoder(tnc_encoder, tnc_classifier, dataloader=valid_loader, train=False)
        else:
            _, test_acc_tnc, test_auc_tnc, test_auprc_tnc, _ = epoch_run_encoder(tnc_encoder, tnc_classifier, dataloader=test_loader, train=False)
   
        tnc_accs.append(test_acc_tnc)
        tnc_aucs.append(test_auc_tnc)
        tnc_auprcs.append(test_auprc_tnc)
 
        with open("./ckpt/%s_classifiers.txt" % data, "a") as f:
            f.write("\n\nPerformance result for a fold")
            f.write("TNC4Maneuvering model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_auc_tnc), str(100 * best_acc_tnc)))
        torch.cuda.empty_cache()
        

    print('=======> Performance Summary:')
    print('TNC4Maneuvering model: \t Accuracy: %.2f +- %.2f \t AUC: %.3f +- %.3f \t AUPRC: %.3f +- %.3f' %
          (100 * np.mean(tnc_accs), 100 * np.std(tnc_accs), np.mean(tnc_aucs), np.std(tnc_aucs), np.mean(tnc_auprcs), np.std(tnc_auprcs)))

if __name__ == '__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run PCA-pruning classification')
    parser.add_argument('--dataset', type=str, default='one_ds')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--window_size', type=int, default=250)
    args = parser.parse_args()

    data_path = './tnc4maneuvering/dataset/'
    if args.dataset == 'one_ds':
        data_path = os.path.join(data_path, 'one_ds/')
    elif args.dataset == 'one_dl':
        data_path = os.path.join(data_path, 'one_dl/')
    elif args.dataset == 'eight_d':
        data_path = os.path.join(data_path, 'eight_d/')
    else:
        print("Incorrect data_type specified.")
        exit()

    if not os.path.exists('./ckpt/classifier'):
        os.mkdir('./ckpt/classifier')

    f = open("./outputs/%s_classifiers.txt"%args.dataset, "w")
    f.close()   
    run_test_pca(data=args.dataset, tnc_lr=0.01, data_path=data_path, window_size=args.window_size, n_cross_val=args.cv) 

end_time = datetime.now()
time_taken = end_time - start_time
print("Overal time taken: ", time_taken) 