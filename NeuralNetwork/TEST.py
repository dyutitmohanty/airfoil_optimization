import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from DATALOADER import AIRFOIL_DATASET
from MODEL import AIRFOIL_GENERATOR

# Reproducibility:
torch.manual_seed(42)


#-------------------------------HYPERPARAMETERS------------------------------#
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 1


#------------------------------CHOOSING DEVICE AND SAVIING--------------------#
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#--------------------------------DATALOADERS:--------------------------------#

TEST_DATASET = AIRFOIL_DATASET(MODE='Test')
TEST_LOADER = DataLoader(TEST_DATASET, BATCH_SIZE, shuffle=True)


#--------------------------------MODEL:--------------------------------#
model = AIRFOIL_GENERATOR()
model = model.to(device)

checkpoint_file = 'checkpt.pth'
checkpoint = torch.load(checkpoint_file)

model.load_state_dict(checkpoint)
model.to(device)

#--------------------------------LOSS:--------------------------------#
loss_fn = torch.nn.HuberLoss(delta=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)


#--------------------------------TESTING:--------------------------------#
test_losses = []

model.eval()


for epoch in tqdm(range(EPOCHS)):
    with torch.no_grad():
        for input,output in TEST_LOADER:
            mini_batch_losses = []

            input = input.to(device, dtype=torch.float32)
            output = output.to(device, dtype=torch.float32)

            predicted_coeff = model(input)

            mini_batch_loss = loss_fn(predicted_coeff,output)
            mini_batch_losses.append(mini_batch_loss.item())

    test_loss = np.mean(mini_batch_losses)
    test_losses.append(test_loss)   

print(test_losses)


