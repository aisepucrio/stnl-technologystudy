import polars as pl #Importando a biblioteca polars
from sklearn.model_selection import train_test_split as tsplit #Importando a biblioteca de divisão de dados
import torch #Importando a biblioteca de tensores
import torch.nn as nn #Importando a biblioteca de redes neurais
import torch.optim as optim #Importando a biblioteca de otimizadores
import torch # Importando a biblioteca de tensores
import numpy as np #Importando a biblioteca numpy
import joblib #Importando a biblioteca joblib

#Lendo os dados
df = pl.read_csv("cancer.csv")
df_resampled = pl.read_csv("df_resampled.csv")

#Dividindo os dados
features = df_resampled.drop(columns=['LUNG_CANCER', 'AGE']).to_numpy()
target = df_resampled['LUNG_CANCER'].to_numpy()

#x_train, x_test, y_train, y_test = tsplit(features, target, test_size = 0.2, random_state = 42)
x_train, x_test, y_train, y_test = tsplit(features, target, test_size=0.2, random_state=None, shuffle=True)

#Definindo a estrutura da rede neural
class LungCancer(nn.Module):
    def __init__(self, input_dim):
        super(LungCancer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64) #Input layer
        self.fc2 = nn.Linear(64, 32) #Hidden layer
        self.fc3 = nn.Linear(32, 16) #Hidden layer
        self.fc4 = nn.Linear(16, 1) #Output layer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Certifique-se de que a dimensão de entrada corresponda ao número de características em X_train
input_dim = x_train.shape[1]
model = LungCancer(input_dim)

# Definir a função de perda
criterion = nn.BCELoss()  # Binary Cross Entropy Loss para classificação binária

# Definir o otimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Certificar que os dados são do tipo float32
X_train = x_train.astype(np.float32)
X_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Convertendo dados de treinamento para tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Convertendo dados de teste para tensores
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Definir o número de épocas
num_epochs = 1000

# Lista para armazenar as perdas
losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
# Salvar o modelo e os objetos de pré-processamento
torch.save(model.state_dict(),'model.pt')
