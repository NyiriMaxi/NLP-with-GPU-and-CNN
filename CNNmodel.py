import torch.nn as nn


class CNNModel(nn.Module):


    def __init__(self, vocab_size, embed_size, num_classes,max_len):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Conv1d(embed_size, 128, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(128 * ((max_len - 5 + 1) // 2), 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1) 
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x