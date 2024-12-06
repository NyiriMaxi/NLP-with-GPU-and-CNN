import matplotlib as plt
import torch as tch
import numpy as np
import os as os
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns

from dataprocess import updated_dataframe,callbackList

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import Checkpoint as chk

from CNNmodel import CNNModel



#Adatok szétosztása

txt_train,txt_test,lng_train,lng_test=train_test_split(updated_dataframe["Text"],updated_dataframe["Language"],test_size=0.2,random_state=42)
lng_coder=LabelEncoder()
#Kódolás
codedLng_train=lng_coder.fit_transform(lng_train)
codedLng_test=lng_coder.transform(lng_test)

#CNN megvalósítás

tokenizer=Tokenizer(num_words=20000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=False)
tokenizer.fit_on_texts(txt_train)

X_trainSeq=tokenizer.texts_to_sequences(txt_train)
X_testSeq=tokenizer.texts_to_sequences(txt_test)
max_Len=max(max(len(x) for x in X_trainSeq),max(len(x) for x in X_testSeq))  #A leghosszabb megtalálása hogy kitöltsük az üres részeket padding segítségével

X_trainPad=pad_sequences(X_trainSeq,maxlen=max_Len)
X_testPad=pad_sequences(X_testSeq,maxlen=max_Len)

# Tensorok létrehozása
X_train_tensor = torch.tensor(X_trainPad, dtype=torch.long)
y_train_tensor = torch.tensor(codedLng_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_testPad, dtype=torch.long)
y_test_tensor = torch.tensor(codedLng_test, dtype=torch.long)

# Adatbetöltők létrehozása
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#model felépítése

    
# Modell létrehozása és áthelyezése GPU-ra
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel(vocab_size=20000, embed_size=300, num_classes=len(lng_coder.classes_),max_len=max_Len).to(device)

#tanítás

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters())
scaler= torch.cuda.amp.GradScaler()

#Ha van elmentett tanítás akkor azt töltjük be

model_path='best_model.pth' 

def train(num_epochs,model,device,optimizer,lossmodel,trainLoader,testLoader):
    train_losses_h = []
    test_accuracies_h = []
    
    best_accuracy = 0.0
    # Tanítási ciklus
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        

        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)  # Áthelyezés GPU-ra

            optimizer.zero_grad()  
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)  
                loss = lossmodel(outputs, labels)  
            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()
              
        epoch_loss = running_loss / len(trainLoader)
        epoch_accuracy = evaluate(cnn_model,testLoader)
        train_losses_h.append(epoch_loss)
        test_accuracies_h.append(epoch_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Veszteség: {epoch_loss:.4f}, Pontosság: {epoch_accuracy:.4f}')

        if(epoch_accuracy>best_accuracy):
            best_accuracy=epoch_accuracy
            chk.save_checkpoint(epoch,model=cnn_model,accuracy=best_accuracy,losses=train_losses_h,test_accuracies=test_accuracies_h,file_path=model_path)
    return train_losses_h,test_accuracies_h

def evaluate(model,testLoader):
    model.eval()
    correct = 0
    total = 0   

    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if(os.path.exists(model_path)):
    print("Mentett modell betöltése...")
    start_epoch, best_accuracy, train_losses, test_accuracies = chk.load_checkpoint(model_path, cnn_model)
else:
    print("Nem volt elmentett modell, tanítás kezdése...")
    train_losses,test_accuracies=train(num_epochs=50,model=cnn_model,device=device,optimizer=optimizer,lossmodel=nn.CrossEntropyLoss(),trainLoader=train_loader,testLoader=test_loader)


# Grafikonok létrehozása
plt.figure(figsize=(12, 5))

# Veszteségi grafikon
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Tanítási veszteség')
plt.title('Tanítási veszteség')
plt.xlabel('Epoch')
plt.ylabel('Veszteség')
plt.legend()

# Pontosság grafikon
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Validációs pontosság', color='orange')
plt.title('Validációs pontosság')
plt.xlabel('Epoch')
plt.ylabel('Pontosság')
plt.legend()

plt.tight_layout()
plt.show()


# A teszt adatok batch-elése 

def Other_Scores():
    y_pred=[]
    cnn_model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)  
            outputs = cnn_model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_pred.append(preds.cpu().numpy()) 

    y_pred = np.concatenate(y_pred)   
    return y_pred
   
    
# Eredmények kiértékelése  
y_pred=Other_Scores()
print('Acc test:', accuracy_score(codedLng_test, y_pred))
print('Precision test:', precision_score(codedLng_test, y_pred, average='macro'))
print('Recall test:', recall_score(codedLng_test, y_pred, average='macro'))
print('F1 test:', f1_score(codedLng_test, y_pred, average='macro'))



def plot_confusion_matrix(ytrue,ypred,classes,title):
    codedClasses=lng_coder.fit_transform(classes)
    cmatrix=confusion_matrix(ytrue,ypred,labels=codedClasses)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Paired', xticklabels=lng_coder.inverse_transform(codedClasses), yticklabels=lng_coder.inverse_transform(codedClasses))
    plt.title(title)
    plt.xlabel('Tippelt nyelv')
    plt.ylabel('Valós nyelv')
    plt.show()

plot_confusion_matrix(codedLng_test,y_pred,lng_coder.classes_,'Konfúziós mátrix')




def testing(texts):
    new_sequences = tokenizer.texts_to_sequences(texts)

    new_padded = pad_sequences(new_sequences, maxlen=max_Len)

    new_tensor=torch.tensor(new_padded, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = cnn_model(new_tensor)
        new_predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    predicted_languages = lng_coder.inverse_transform(new_predictions)

    
    return predicted_languages[0]

    


