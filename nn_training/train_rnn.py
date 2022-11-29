import tqdm
import torch
import os
import math
import numpy as np
from modules.constants import *
from modules.dataset import FittingDataset
from models.rnn import FittingRNN
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from timeit import default_timer as timer

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# training dataset
dataset = FittingDataset(TRAINING_DATASET)

# function to collate data samples into batched tensors
def collate_fn(batch, padding_value = 0):
    (xx, yy) = zip(*batch)
    
    # remove None
    xx = [x for x in xx if x is not None]
    yy = [y for y in yy if y is not None]
    
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=padding_value)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=padding_value)

    return xx_pad, yy_pad, x_lens, y_lens

# training function (to be called per epoch)
def train_epoch(model, optimizer, disable=False):
    model.train()
    losses = 0.
    n_batches = int(math.ceil(len(train_loader.dataset)/BATCH_SIZE)) #if max_iters_train is None else max_iters_train
    t = tqdm.tqdm(enumerate(train_loader),total=n_batches, disable=disable)
    for i, data in t:
        src, tgt, src_len, tgt_len = data
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
    
        # run model
        preds = model.forward(src, src_len)
        
        optimizer.zero_grad()
        
        # create a mask by filtering out all tokens that ARE NOT the padding token   
        mask = (tgt>=0).float()
        tgt = tgt * mask
        preds = preds * mask
        
        # loss calculation
        pred_packed = torch.nn.utils.rnn.pack_padded_sequence(preds, src_len, batch_first=True, enforce_sorted=False)
        tgt_packed = torch.nn.utils.rnn.pack_padded_sequence(tgt, src_len, batch_first=True, enforce_sorted=False)
        loss = loss_fn(pred_packed.data, tgt_packed.data)
        loss.backward() # compute gradients
        
        t.set_description("loss = %.7f" % loss.item() )
        
        optimizer.step() # backprop
        losses += loss.item()
          
    return losses / len(train_loader)

# test function
def evaluate(model, disable):
    model.eval()
    losses = 0
    n_batches = int(math.ceil(len(valid_loader.dataset)/BATCH_SIZE))
    t = tqdm.tqdm(enumerate(valid_loader),total=n_batches, disable=disable)

    with torch.no_grad():
        for i, data in t:
            src, tgt, src_len, tgt_len = data
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            # create masks
            src_mask, src_padding_mask = create_mask_src(src)

            # run model
            preds = model(src, src_mask, src_padding_mask, src_padding_mask)

            # create a mask by filtering out all tokens that ARE NOT the padding token    
            mask = (tgt != PAD_IDX).float()
            tgt = tgt * mask
            preds = preds * mask

            # loss calculation
            pred_packed = torch.nn.utils.rnn.pack_padded_sequence(preds, src_len, batch_first=True, enforce_sorted=False)
            tgt_packed = torch.nn.utils.rnn.pack_padded_sequence(tgt, src_len, batch_first=True, enforce_sorted=False)
            loss = loss_fn(pred_packed.data, tgt_packed.data)
            losses += loss.item()

    return losses / len(valid_loader)

# split dataset into training and validation sets
fulllen = len(dataset)
train_len = int(fulllen*0.8)
val_len = fulllen-train_len
train_set, val_set, = random_split(dataset, [train_len, val_len],
                                            generator=torch.Generator().manual_seed(7))
train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=BATCH_SIZE,\
                          num_workers=4, shuffle=True)
valid_loader = DataLoader(val_set, collate_fn=collate_fn, batch_size=BATCH_SIZE,\
                          num_workers=4, shuffle=False)

torch.manual_seed(7) # for reproducibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
rnn = FittingRNN(nb_layers=5, sum_outputs=True, rnn="gru", nb_lstm_units=50, input_size=4,\
                   output_size=3, batch_size=BATCH_SIZE, dropout=0.1, bidirectional=True,\
                   learn_init_states=False, init_states="rand", device=DEVICE)
rnn = rnn.to(DEVICE)
print(rnn)

pytorch_total_params = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
print("Total trainable params: {}".format(pytorch_total_params))

# loss and optimiser
loss_fn = torch.nn.MSELoss() # MSE loss
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPS)

train_losses, val_losses = [], []
min_val_loss = np.inf
disable, load = False, False
epoch, count = 0, 0

if load:
    print("Loading saved model...")
    checkpoint = torch.load("models/rnn_last")
    transformer.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']+1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    min_val_loss = min(val_losses)
    count = checkpoint['count']
    print(epoch, val_losses)
else:
    print("Starting training...")

for epoch in range(epoch, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(rnn, optimizer, disable)
    end_time = timer()
    val_loss = evaluate(rnn, disable)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, Val loss: {val_loss:.8f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        print("Saving best model with val_loss: {}".format(val_loss))
        torch.save({
                   'epoch': epoch,
                   'model_state_dict': transformer.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'train_losses': train_losses,
                   'val_losses': val_losses,
                   'count': count,
                   }, "models/rnn_best")
        count=0
    else:
        print("Saving last model with val_loss: {}".format(val_loss))
        torch.save({
                   'epoch': epoch,
                   'model_state_dict': transformer.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'train_losses': train_losses,
                   'val_losses': val_losses,
                   'count': count,
                   }, "models/rnn_last")
        count+=1

    if count>=EARLY_STOPPING:
        print("Early stopping...")
        brea

train_losses, val_losses = [], []
min_val_loss = np.inf
disable, load = False, False
epoch, count = 0, 0

if load:
    print("Loading saved model...")
    checkpoint = torch.load("models/rnn_last")
    transformer.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']+1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    min_val_loss = min(val_losses)
    count = checkpoint['count']
    print(epoch, val_losses)
else:
    print("Starting training...")

for epoch in range(epoch, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(rnn, optimizer, disable)
    end_time = timer()
    val_loss = evaluate(rnn, disable)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.8f}, Val loss: {val_loss:.8f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        print("Saving best model with val_loss: {}".format(val_loss))
        torch.save({
                   'epoch': epoch,
                   'model_state_dict': transformer.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'train_losses': train_losses,
                   'val_losses': val_losses,
                   'count': count,
                   }, "models/rnn_best")
        count=0
    else:
        print("Saving last model with val_loss: {}".format(val_loss))
        torch.save({
                   'epoch': epoch,
                   'model_state_dict': transformer.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'train_losses': train_losses,
                   'val_losses': val_losses,
                   'count': count,
                   }, "models/rnn_last")
        count+=1

    if count>=EARLY_STOPPING:
        print("Early stopping...")
        break
