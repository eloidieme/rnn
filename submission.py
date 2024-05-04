import numpy as np
import torch
import torch.nn.functional as F
import pickle
import time
import random

book_fname = "./data/goblet_book.txt"
with open(book_fname, 'r') as book:
    book_data = book.read()
len(book_data)

word_list = book_data.split()
chars = [[*word] for word in word_list]
max_len = max(len(word) for word in chars)
for wordl in chars:
    while len(wordl) < max_len:
        wordl.append(' ')
chars = np.array(chars)

unique_chars = list(np.unique(chars))
unique_chars.append('\n')
unique_chars.append('\t')
K = len(unique_chars)  # dimensionality of the input and output vectors

char_to_ind = {}
ind_to_char = {}
for idx, char in enumerate(unique_chars):
    char_to_ind[char] = idx
    ind_to_char[idx] = char

m = 100  # dimensionality of the hidden state
eta = 0.1  # learning rate
seq_length = 25  # length of input sequences used during training
epsilon = 1e-8  # for AdaGrad

sig = 0.01
RNN = {'b': torch.zeros((m, 1), dtype=torch.double), 'c': torch.zeros((K, 1), dtype=torch.double), 'U': torch.normal(0.0, sig, (m, K), dtype=torch.double), 'W': torch.normal(0.0, sig, (m, m), dtype=torch.double), 'V': torch.normal(0.0, sig, (K, m), dtype=torch.double), 'h0': torch.zeros((m, 1), dtype=torch.double)}

def encode_char(char):
    oh = [0]*K
    oh[char_to_ind[char]] = 1
    return oh

def synthetize_seq(rnn, h0, x0, n, T = 1):
    t, ht, xt = 0, h0, x0
    indexes = []
    while t < n:
        xt = xt.reshape((K, 1))
        at = torch.mm(rnn['W'], ht) + torch.mm(rnn['U'], xt) + rnn['b']
        ht = torch.tanh(at)
        ot = torch.mm(rnn['V'], ht) + rnn['c']
        pt = F.softmax(ot/T, dim=0)
        cp = torch.cumsum(pt, dim=0)
        a = torch.rand(1)
        ixs = torch.where(cp - a > 0)
        ii = ixs[0][0].item()
        indexes.append(ii)
        xt = torch.zeros((K, 1), dtype=torch.double)
        xt[ii, 0] = 1
        t += 1
    Y = []
    for idx in indexes:
        oh = [0]*K
        oh[idx] = 1
        Y.append(oh)
    Y = torch.tensor(Y).t()
    
    s = ''
    for i in range(Y.shape[1]):
        idx = torch.where(Y[:, i] == 1)[0].item()
        s += ind_to_char[idx]
    
    return Y, s

def encode_string(chars):
    M = []
    for i in range(len(chars)):
        M.append(encode_char(chars[i]))
    M = torch.tensor(M, dtype=torch.double).t()
    return M

def forward(rnn, X, hprev):
    ht = hprev.clone()
    indexes = []
    P = torch.zeros((K, seq_length), dtype=torch.double)
    A = torch.zeros((m, seq_length), dtype=torch.double)
    H = torch.zeros((m, seq_length), dtype=torch.double)
    for i in range(seq_length):
        xt = X[:, i].reshape((K, 1))
        at = torch.mm(rnn['W'], ht) + torch.mm(rnn['U'], xt) + rnn['b']
        ht = torch.tanh(at)
        ot = torch.mm(rnn['V'], ht) + rnn['c']
        pt = F.softmax(ot, dim=0)

        H[:, i] = ht.squeeze()
        P[:, i] = pt.squeeze()
        A[:, i] = at.squeeze()
        cp = torch.cumsum(pt, dim=0)
        a = torch.rand(1)
        ixs = torch.where(cp - a > 0)
        ii = ixs[0][0].item()
        indexes.append(ii)

    Y_pred = []
    for idx in indexes:
        oh = [0]*K
        oh[idx] = 1
        Y_pred.append(oh)
    Y_pred = torch.tensor(Y_pred, dtype=torch.double).t()

    s_pred = ''
    for i in range(Y_pred.shape[1]):
        idx = torch.where(Y_pred[:, i] == 1)[0].item()
        s_pred += ind_to_char[idx]

    return s_pred, Y_pred, A, H, P, ht

def compute_loss(Y, P):
    log_probs = torch.log(P)
    cross_entropy = -torch.sum(Y * log_probs)
    loss = cross_entropy.item()
    return loss

def backward(rnn, X, Y, A, H, P, hprev):
    dA = torch.zeros_like(A)
    dH = torch.zeros_like(H)

    G = -(Y - P)
    dV = torch.matmul(G, H.t())
    dhtau = torch.matmul(G[:, -1], rnn['V'])
    datau = (1 - torch.pow(torch.tanh(A[:, -1]), 2)) * dhtau
    dH[:, -1] = dhtau.squeeze()
    dA[:, -1] = datau.squeeze()

    for i in range(seq_length - 2, -1, -1):
        dht = torch.matmul(G[:, i], rnn['V']) + torch.matmul(dA[:, i+1].reshape(1, -1), rnn['W'])
        dat = (1 - torch.pow(torch.tanh(A[:, i]), 2)) * dht
        dH[:, i] = dht.squeeze()
        dA[:, i] = dat.squeeze()

    Hd = torch.cat((hprev, H[:, :-1]), dim=1)
    dW = torch.matmul(dA, Hd.t())
    dU = torch.matmul(dA, X.t())
    dc = G.sum(1).reshape((-1, 1))
    db = dA.sum(1).reshape((-1, 1))
    grads = {'U': dU, 'W': dW, 'V': dV, 'c': dc, 'b': db}
    grads_clamped = {k: torch.clamp(v, min=-5.0, max=5.0) for (k,v) in grads.items()}
    return grads, grads_clamped

def ComputeGradNum(X, Y, param_name, rnn, h=1e-4):
    """
    Compute the numerical gradient of the rnn's parameter specified by param_name.
    """
    grad = torch.zeros_like(rnn[param_name])
    hprev = rnn['h0']
    n = torch.numel(rnn[param_name])
    
    for i in range(n):
        old_val = rnn[param_name].view(-1)[i].item()
        rnn[param_name].view(-1)[i] = old_val - h
        _, _, _, _, P, _ = forward(rnn, X, hprev)
        l1 = compute_loss(Y, P)
        
        rnn[param_name].view(-1)[i] = old_val + h
        _, _, _, _, P, _ = forward(rnn, X, hprev)
        l2 = compute_loss(Y, P)
        
        grad.view(-1)[i] = (l2 - l1) / (2 * h)
        rnn[param_name].view(-1)[i] = old_val  # Reset to original value

    return grad

def ComputeGradsNum(X, Y, rnn, h=1e-4):
    num_grads = {}
    for param_name in rnn:
        if param_name != 'h0':
            print('Computing numerical gradient for')
            print(f'Field name: {param_name}')
            num_grads[param_name] = ComputeGradNum(X, Y, param_name, rnn, h)
    return num_grads

############################# GRADIENT CHECK #############################  
X_chars = book_data[:seq_length]
Y_chars = book_data[1:seq_length+1]
X = encode_string(X_chars)
Y = encode_string(Y_chars)
s_pred, Y_pred, A, H, P, ht = forward(RNN, X, RNN['h0'])
grads, grads_clamped = backward(RNN, X, Y, A, H, P, RNN['h0'])
num_grads = ComputeGradsNum(X, Y, RNN, 1e-4)
print("-------- Gradient validation --------")
print("Max diff for gradient of b:", torch.max(torch.abs(num_grads['b'] - grads['b'])).item())
print("Max diff for gradient of c:", torch.max(torch.abs(num_grads['c'] - grads['c'])).item())
print("Max diff for gradient of W:", torch.max(torch.abs(num_grads['W'] - grads['W'])).item())
print("Max diff for gradient of U:", torch.max(torch.abs(num_grads['U'] - grads['U'])).item())
print("Max diff for gradient of V:", torch.max(torch.abs(num_grads['V'] - grads['V'])).item())
##########################################################################

############################# TRAINING - AdaGrad ############################# 
e, step, epoch = 0, 0, 0
n_epochs = 10
smooth_loss = 0
losses = []
hprev = RNN['h0']

mb = torch.zeros_like(RNN['b'], dtype=torch.float)
mc = torch.zeros_like(RNN['c'], dtype=torch.float)
mU = torch.zeros_like(RNN['U'], dtype=torch.float)
mV = torch.zeros_like(RNN['V'], dtype=torch.float)
mW = torch.zeros_like(RNN['W'], dtype=torch.float)
ms = {'b': mb, 'c': mc, 'U': mU, 'V': mV, 'W': mW}

while epoch < n_epochs:
    X_chars = book_data[e:e+seq_length]
    Y_chars = book_data[e+1:e+seq_length+1]
    X_train = encode_string(X_chars)
    Y_train = encode_string(Y_chars)

    _, _, A_train, H_train, P_train, ht = forward(RNN, X_train, hprev)
    loss = compute_loss(Y_train, P_train)
    grads, grads_clamped = backward(RNN, X_train, Y_train, A_train, H_train, P_train, hprev)

    for k in ms.keys():
        ms[k] += grads_clamped[k]**2
        RNN[k] -= (eta/torch.sqrt(ms[k] + epsilon))*grads_clamped[k]

    if step == 0:
        smooth_loss = loss
    else:
        smooth_loss = 0.999*smooth_loss + 0.001*loss

    losses.append(smooth_loss)

    if step % 1000 == 0:
        print(f"Step: {step}")
        print(f"\t * Smooth loss: {smooth_loss}")
    if step % 5000 == 0:
        _, s_syn = synthetize_seq(RNN, hprev, X_train[:, 0], 200, 0.6)
        print("-" * 100)
        print(f"Synthetized sequence: \n{s_syn}")
        print("-" * 100)
    if step % 100000 == 0 and step > 0:
        _, s_lsyn = synthetize_seq(RNN, hprev, X_train[:, 0], 1000, 0.6)
        print("-" * 100)
        print(f"Long synthetized sequence: \n{s_lsyn}")
        print("-" * 100)

    step += 1
    e += seq_length
    if e > len(book_data) - seq_length:
        e = 0
        epoch += 1
        hprev = RNN['h0']
    else:
        hprev = ht

with open(f'rnn_{time.time()}.pickle', 'wb') as handle:
    pickle.dump(RNN, handle, protocol=pickle.HIGHEST_PROTOCOL)
##############################################################################

import matplotlib.pyplot as plt

plt.grid(True)
plt.plot(losses)
plt.xlabel('Steps')
plt.ylabel('Smooth loss')
plt.title(f'Training with AdaGrad - eta: {eta} - seq_length: {seq_length} - m: {m} - n_epochs: {n_epochs}')
plt.savefig('./training.png')

with open('rnn.pickle', 'rb') as handle:
    test_rnn = pickle.load(handle)

Y_t, s_t = synthetize_seq(test_rnn, test_rnn['h0'], X[:,0], 1000, 0.6)
print(s_t)

############################# TRAINING - Adam #############################
e, step, epoch = 0, 0, 0
n_epochs = 10
smooth_loss = 0
losses = []
hprev = RNN['h0']

beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8

mb = torch.zeros_like(RNN['b'], dtype=torch.float)
vb = torch.zeros_like(RNN['b'], dtype=torch.float)
mc = torch.zeros_like(RNN['c'], dtype=torch.float)
vc = torch.zeros_like(RNN['c'], dtype=torch.float)
mU = torch.zeros_like(RNN['U'], dtype=torch.float)
vU = torch.zeros_like(RNN['U'], dtype=torch.float)
mV = torch.zeros_like(RNN['V'], dtype=torch.float)
vV = torch.zeros_like(RNN['V'], dtype=torch.float)
mW = torch.zeros_like(RNN['W'], dtype=torch.float)
vW = torch.zeros_like(RNN['W'], dtype=torch.float)
ms = {'b': mb, 'c': mc, 'U': mU, 'V': mV, 'W': mW}
vs = {'b': vb, 'c': vc, 'U': vU, 'V': vV, 'W': vW}

while epoch < n_epochs:
    X_chars = book_data[e:e+seq_length]
    Y_chars = book_data[e+1:e+seq_length+1]
    X_train = encode_string(X_chars)
    Y_train = encode_string(Y_chars)

    _, _, A_train, H_train, P_train, ht = forward(RNN, X_train, hprev)
    loss = compute_loss(Y_train, P_train)
    grads, grads_clamped = backward(RNN, X_train, Y_train, A_train, H_train, P_train, hprev)

    for k in ms.keys():
        ms[k] = beta_1*ms[k] + (1 - beta_1)*grads_clamped[k]
        vs[k] = beta_2*vs[k] + (1 - beta_2)*(grads_clamped[k]**2)
        m_hat = ms[k]/(1 - beta_1**(step+1))
        v_hat = vs[k]/(1 - beta_2**(step+1))
        RNN[k] -= (eta/torch.sqrt(v_hat + epsilon))*m_hat

    if step == 0:
        smooth_loss = loss
    else:
        smooth_loss = 0.999*smooth_loss + 0.001*loss

    losses.append(smooth_loss)

    if step % 1000 == 0:
        print(f"Step: {step}")
        print(f"\t * Smooth loss: {smooth_loss}")
    if step % 5000 == 0:
        _, s_syn = synthetize_seq(RNN, hprev, X_train[:, 0], 200)
        print("-" * 100)
        print(f"Synthetized sequence: \n{s_syn}")
        print("-" * 100)
    if step % 100000 == 0 and step > 0:
        _, s_lsyn = synthetize_seq(RNN, hprev, X_train[:, 0], 1000)
        print("-" * 100)
        print(f"Long synthetized sequence: \n{s_lsyn}")
        print("-" * 100)

    step += 1
    e += seq_length
    if e > len(book_data) - seq_length:
        e = 0
        epoch += 1
        hprev = RNN['h0']
    else:
        hprev = ht

with open(f'rnn_{time.time()}.pickle', 'wb') as handle:
    pickle.dump(RNN, handle, protocol=pickle.HIGHEST_PROTOCOL)
###########################################################################

############################# TRAINING - Chunks #############################
def split_into_chunks(s, L):
    chunk_size = len(s) // L
    remainder = len(s) % L

    chunks = []
    for i in range(L):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(s[start:end])
    return chunks

step, epoch = 0, 0
n_epochs = 10
smooth_loss = 0
losses = []

mb = torch.zeros_like(RNN['b'], dtype=torch.float)
mc = torch.zeros_like(RNN['c'], dtype=torch.float)
mU = torch.zeros_like(RNN['U'], dtype=torch.float)
mV = torch.zeros_like(RNN['V'], dtype=torch.float)
mW = torch.zeros_like(RNN['W'], dtype=torch.float)
ms = {'b': mb, 'c': mc, 'U': mU, 'V': mV, 'W': mW}

while epoch < n_epochs:
    print(f"Epoch {epoch+1}/{n_epochs}")
    L = random.randint(10, 20)
    print(f"\t * No. chunks: {L}")
    chunks = split_into_chunks(book_data, L)
    random.shuffle(chunks)
    for idx, chunk in enumerate(chunks):
        print(f"-> Reached chunk {idx+1}")
        e = 0
        hprev = torch.zeros((m, 1), dtype=torch.double)
        while e < (len(chunk) - seq_length):
            X_chars = chunk[e:e+seq_length]
            Y_chars = chunk[e+1:e+seq_length+1]
            X_train = encode_string(X_chars)
            Y_train = encode_string(Y_chars)

            A_train, H_train, P_train, ht = forward(RNN, X_train, hprev)
            loss = compute_loss(Y_train, P_train)
            grads, grads_clamped = backward(RNN, X_train, Y_train, A_train, H_train, P_train, hprev)

            for k in ms.keys():
                ms[k] += grads_clamped[k]**2
                RNN[k] -= (eta/torch.sqrt(ms[k] + epsilon))*grads_clamped[k]

            if step == 0:
                smooth_loss = loss
            else:
                smooth_loss = 0.999*smooth_loss + 0.001*loss

            losses.append(smooth_loss)

            e += seq_length
            hprev = ht

            if step % 1000 == 0:
                print(f"Step: {step}")
                print(f"\t * Smooth loss: {smooth_loss:.4f}")
            if step % 5000 == 0:
                _, s_syn = synthetize_seq(RNN, hprev, X_train[:, 0], 200)
                print("-" * 100)
                print(f"Synthetized sequence: \n{s_syn}")
                print("-" * 100)
            if step % 100000 == 0 and step > 0:
                _, s_lsyn = synthetize_seq(RNN, hprev, X_train[:, 0], 1000)
                print("-" * 100)
                print(f"Long synthetized sequence: \n{s_lsyn}")
                print("-" * 100)
            step += 1
            

    epoch += 1

with open(f'rnn_{time.time()}.pickle', 'wb') as handle:
    pickle.dump(RNN, handle, protocol=pickle.HIGHEST_PROTOCOL)
#############################################################################

############################# TRAINING - Batches ############################

def forward_batch(rnn, X, hprev):
    K, seq_length, batch_size = X.shape
    m = hprev.shape[0]  # (m, batch_size)

    P = torch.zeros((K, seq_length, batch_size), dtype=torch.double)
    A = torch.zeros((m, seq_length, batch_size), dtype=torch.double)
    H = torch.zeros((m, seq_length, batch_size), dtype=torch.double)

    ht = hprev.clone()
    for i in range(seq_length):
        xt = X[:, i, :]  # Access the ith timestep across all batches
        at = torch.mm(rnn['W'], ht) + torch.mm(rnn['U'], xt) + rnn['b'].expand(m, batch_size)
        ht = torch.tanh(at)
        ot = torch.mm(rnn['V'], ht) + rnn['c'].expand(K, batch_size)
        pt = F.softmax(ot, dim=0)

        H[:, i, :] = ht
        P[:, i, :] = pt
        A[:, i, :] = at

    return A, H, P, ht

def compute_loss_batch(Y, P):
    batch_size = Y.shape[2]
    log_probs = torch.log(P)
    cross_entropy = -torch.sum(Y * log_probs)
    loss = cross_entropy.item() / batch_size
    return loss

def backward_batch(rnn, X, Y, A, H, P, hprev):
    dA = torch.zeros_like(A)
    dH = torch.zeros_like(H)

    G = -(Y - P)
    dV = torch.bmm(G.permute(2, 0, 1), H.permute(2, 1, 0)).mean(dim=0)
    dhtau = torch.matmul(G[:, -1, :].t(), rnn['V']).t()
    datau = (1 - torch.pow(torch.tanh(A[:, -1, :]), 2)) * dhtau
    dH[:, -1, :] = dhtau
    dA[:, -1, :] = datau

    for i in range(seq_length - 2, -1, -1):
        dht = torch.matmul(G[:, i, :].t(), rnn['V']).t() + torch.matmul(dA[:, i+1, :].t(), rnn['W']).t()
        dat = (1 - torch.pow(torch.tanh(A[:, i]), 2)) * dht
        dH[:, i] = dht
        dA[:, i] = dat

    Hd = torch.cat((hprev.reshape((m, 1, -1)), H[:, :-1, :]), dim=1)
    dW = torch.matmul(dA.permute(2, 0, 1), Hd.permute(2, 1, 0)).mean(dim=0)
    dU = torch.matmul(dA.permute(2, 0, 1), X.permute(2, 1, 0)).mean(dim=0)
    dc = G.sum(1).mean(dim=1).reshape((-1, 1))
    db = dA.sum(1).mean(dim=1).reshape((-1, 1))
    grads = {'U': dU, 'W': dW, 'V': dV, 'c': dc, 'b': db}
    grads_clamped = {k: torch.clamp(v, min=-5.0, max=5.0) for (k,v) in grads.items()}
    return grads, grads_clamped

e, step, epoch = 0, 0, 0
n_epochs = 10
smooth_loss = 0
batch_size = 10
eta = 0.1
losses = []
hprev = torch.zeros((m, batch_size), dtype=torch.double)

mb = torch.zeros_like(RNN['b'], dtype=torch.float)
mc = torch.zeros_like(RNN['c'], dtype=torch.float)
mU = torch.zeros_like(RNN['U'], dtype=torch.float)
mV = torch.zeros_like(RNN['V'], dtype=torch.float)
mW = torch.zeros_like(RNN['W'], dtype=torch.float)
ms = {'b': mb, 'c': mc, 'U': mU, 'V': mV, 'W': mW}

while epoch < n_epochs:
    X_batch = []
    Y_batch = []
    for b in range(batch_size):
        start_index = e + b * seq_length
        X_chars = book_data[start_index:(start_index + seq_length)]
        Y_chars = book_data[(start_index + 1):(start_index + seq_length + 1)]
        X_batch.append(encode_string(X_chars))
        Y_batch.append(encode_string(Y_chars))

    X_train = torch.stack(X_batch, dim=2)  # shape: (K, seq_length, batch_size)
    Y_train = torch.stack(Y_batch, dim=2)  # shape: (K, seq_length, batch_size)

    A_train, H_train, P_train, hts = forward_batch(RNN, X_train, hprev)
    loss = compute_loss_batch(Y_train, P_train)
    grads, grads_clamped = backward_batch(RNN, X_train, Y_train, A_train, H_train, P_train, hprev)

    for k in ms.keys():
        ms[k] += grads_clamped[k]**2
        RNN[k] -= (eta/torch.sqrt(ms[k] + epsilon)) * grads_clamped[k]

    if step == 0:
        smooth_loss = loss
    else:
        smooth_loss = 0.999*smooth_loss + 0.001*loss
    losses.append(smooth_loss)

    if step % 1000 == 0:
        print(f"Step: {step}")
        print(f"\t * Smooth loss: {smooth_loss}")
    if step % 5000 == 0:
        _, s_syn = synthetize_seq(RNN, hprev[:, 0:1], X_train[:, 0, 0], 200, 0.6)
        print("-" * 100)
        print(f"Synthetized sequence: \n{s_syn}")
        print("-" * 100)
    if step % 100000 == 0 and step > 0:
        _, s_lsyn = synthetize_seq(RNN, hprev[:, 0:1], X_train[:, 0, 0], 1000, 0.6)
        print("-" * 100)
        print(f"Long synthetized sequence: \n{s_lsyn}")
        print("-" * 100)

    step += 1
    e += batch_size * seq_length
    if e > len(book_data) - batch_size * seq_length:
        e = 0
        epoch += 1
        hprev = torch.zeros((m, batch_size), dtype=torch.double)
    else:
        hprev = hts

with open(f'rnn_{time.time()}.pickle', 'wb') as handle:
    pickle.dump(RNN, handle, protocol=pickle.HIGHEST_PROTOCOL)
#############################################################################

############################# Nucleus sampling #############################
def synthetize_seq(rnn, h0, x0, n, theta = 0.8):
    t, ht, xt = 0, h0, x0
    indexes = []
    while t < n:
        xt = xt.reshape((K, 1))
        at = torch.mm(rnn['W'], ht) + torch.mm(rnn['U'], xt) + rnn['b']
        ht = torch.tanh(at)
        ot = torch.mm(rnn['V'], ht) + rnn['c']
        pt = F.softmax(ot, dim=0)
        sorted_probs, sorted_indices = torch.sort(pt, descending=True, dim=0)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff_index = (cumulative_probs >= theta).nonzero().min().item()
        valid_probs = sorted_probs[:cutoff_index + 1]
        valid_indices = sorted_indices[:cutoff_index + 1]
        rescaled_probs = valid_probs / valid_probs.sum()
        char_index = valid_indices[torch.multinomial(rescaled_probs, 1)]
        indexes.append(char_index.item())
        xt = torch.zeros((K, 1), dtype=torch.double)
        xt[char_index, 0] = 1
        t += 1
    Y = []
    for idx in indexes:
        oh = [0]*K
        oh[idx] = 1
        Y.append(oh)
    Y = torch.tensor(Y).t()
    
    s = ''
    for i in range(Y.shape[1]):
        idx = torch.where(Y[:, i] == 1)[0].item()
        s += ind_to_char[idx]
    
    return Y, s