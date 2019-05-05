import torch 
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import data_helpers_v2
import torch.nn.functional as F


X_train,Y_train,X_test,Y_test,vocabulary, vocabulary_inv_list,embedding_matrix = data_helpers_v2.loads_data()


#embedding_matrix = torch.LongTensor(embedding_matrix)


print("Embeddings ",embedding_matrix.shape)

#X =  torch.LongTensor(X)
#Y = torch.LongTensor(Y)

X_train =  torch.LongTensor(X_train)
Y_train = torch.LongTensor(Y_train)

X_test =  torch.LongTensor(X_test)
Y_test = torch.LongTensor(Y_test)



train  = data.TensorDataset(X_train,Y_train)
train_loader = data.DataLoader(train)

test  = data.TensorDataset(X_test,Y_test)
test_loader = data.DataLoader(test)


# Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device = "cpu"

# Hyper-parameters
sequence_length = 80
input_size = 300
hidden_size = 128
num_layers = 2
num_classes = 2
batch_size = 32
num_epochs = 30
learning_rate = 0.001




# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,vocab_size,num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(vocab_size,input_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        x = self.embedding(x)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out = self.dropout(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        #out = F.softmax(out,dim=0)
        return out

model = BiRNN(input_size, hidden_size, num_layers,len(vocabulary),num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        #images = images.reshape(-1, sequence_length, input_size).to(device)
        #labels = labels.to(device)
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    
        if (i+1) % 500 == 0:
            
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 



# Test the model

    with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in test_loader:
         images = images.to(device)
         labels = labels.to(device)
         outputs = model(images)
         _, predicted = torch.max(outputs.data, 1)
         total += labels.size(0)
         correct += (predicted == labels).sum().item()

      print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')
