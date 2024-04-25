import torch.nn.functional as F
import torch

"""
Manual gradient computation for a RNN that predicts the next character
in a character sequence.
"""

K = 3  # number of possible characters in sequence
seq_length = 4  # length of input sequences used during training
m = 2  # dimensionality of the hidden state
sig = 0.01  # std deviation for params init

h0 = torch.zeros((m, 1), dtype=torch.float, requires_grad=False)

b = torch.zeros((m, 1), dtype=torch.float, requires_grad=True)

c = torch.zeros((K, 1), dtype=torch.float, requires_grad=True)
W = torch.normal(0.0, sig, (m, m), dtype=torch.float, requires_grad=True)
U = torch.normal(0.0, sig, (m, K), dtype=torch.float, requires_grad=True)
V = torch.normal(0.0, sig, (K, m), dtype=torch.float, requires_grad=True)

# Want to learn the word radar
X = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float) # one-hot encoding of 'rada' ; column-major ordering
Y = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]], dtype=torch.float) # one-hot encoding of 'adar' ; column-major ordering

########

P = torch.zeros((K, seq_length), dtype=torch.float)
A = torch.zeros((m, seq_length), dtype=torch.float)
H = torch.zeros((m, seq_length), dtype=torch.float)

#############################################

ht = h0.clone()
for i in range(seq_length):
    xt = X[:, i].reshape((K, 1))
    at = torch.mm(W, ht) + torch.mm(U, xt) + b
    ht = torch.tanh(at)
    ot = torch.mm(V, ht) + c
    pt = F.softmax(ot, dim=0)

    H[:, i] = ht.squeeze()
    P[:, i] = pt.squeeze()
    A[:, i] = at.squeeze()

log_probs = torch.log(P)
cross_entropy = -torch.sum(Y * log_probs)
average_loss = cross_entropy
loss = average_loss.item()
average_loss.backward()

#############################################

dA = torch.zeros_like(A)
dH = torch.zeros_like(H)

G = -(Y - P)
dV = torch.matmul(G, H.t())    ### dV wrong -> there is a problem above
dhtau = torch.matmul(G[:, -1], V)
datau = (1 - torch.pow(torch.tanh(A[:, -1]), 2)) * dhtau
dH[:, -1] = dhtau.squeeze()
dA[:, -1] = datau.squeeze()

for i in range(seq_length - 2, -1, -1):
    dht = torch.matmul(G[:, i], V) + torch.matmul(dA[:, i+1].reshape(1, -1), W)
    dat = (1 - torch.pow(torch.tanh(A[:, i]), 2)) * dht
    dH[:, i] = dht.squeeze()
    dA[:, i] = dat.squeeze()

Hd = torch.cat((h0, H[:, :-1]), dim=1)
dW = torch.matmul(dA, Hd.t())
dU = torch.matmul(dA, X.t())
dc = G.sum(1).reshape((-1, 1))
db = dA.sum(1).reshape((-1, 1))

#############################################

print("-------- Gradient validation --------")
print("Max diff for gradient of b:", torch.max(torch.abs(b.grad - db)).item())
print("Max diff for gradient of c:", torch.max(torch.abs(c.grad - dc)).item())
print("Max diff for gradient of W:", torch.max(torch.abs(W.grad - dW)).item())
print("Max diff for gradient of U:", torch.max(torch.abs(U.grad - dU)).item())
print("Max diff for gradient of V:", torch.max(torch.abs(V.grad - dV)).item())