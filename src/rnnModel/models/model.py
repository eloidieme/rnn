import torch
import torch.nn.functional as F
from rnnModel.data.make_dataset import make_charmap
from tqdm import tqdm

### FOR TESTING PURPOSES ###
M = 100
ETA = 0.1
SEQ_LENGTH = 25
K = 80
############################

class RNN:
    def __init__(self, m = 100, eta = 0.1, epsilon = 1e-8, seq_length = 25, data_path = "./data/goblet_book.txt", autograd = False) -> None:
        ## Constants
        self.m = m  # dimensionality of the hidden state
        self.eta = eta  # learning rate
        self.epsilon = epsilon  # for AdaGrad
        self.seq_length = seq_length  # length of input sequences used during training
        self.book_data, self.K, self.char_to_ind, self.ind_to_char = make_charmap(data_path)  # dimensionality of input and character mappings

        ## Parameters
        # bias vectors
        self.b = torch.zeros((self.m, 1), requires_grad=autograd, dtype=torch.float,)
        self.c = torch.zeros((self.K, 1), requires_grad=autograd, dtype=torch.float,)
        # weight matrices
        sig = 0.01
        self.U = torch.normal(0.0, sig, (self.m, self.K), requires_grad=autograd, dtype=torch.float,)
        self.W = torch.normal(0.0, sig, (self.m, self.m), requires_grad=autograd, dtype=torch.float,)
        self.V = torch.normal(0.0, sig, (self.K, self.m), requires_grad=autograd, dtype=torch.float,)
        self.h0 = torch.zeros((self.m, 1), requires_grad=False, dtype=torch.float,)
        self.params = {
            'W': self.W, 
            'U': self.U,
            'V': self.V,
            'b': self.b,
            'c': self.c
        }

        ## Labeled sequence for debugging
        X_chars = self.book_data[0:seq_length]
        Y_chars = self.book_data[1:seq_length+1]
        self.X = self.encode_string(X_chars)
        self.Y = self.encode_string(Y_chars)

    def forward(self, X, Y, hprev):
        ht = hprev.clone()
        indexes = []
        P = torch.zeros((self.K, self.seq_length), dtype=torch.float)
        A = torch.zeros((self.m, self.seq_length), dtype=torch.float)
        H = torch.zeros((self.m, self.seq_length), dtype=torch.float)
        for i in range(self.seq_length):
            xt = X[:, i].reshape((self.K, 1))
            at = torch.mm(self.W, ht) + torch.mm(self.U, xt) + self.b
            ht = torch.tanh(at)
            ot = torch.mm(self.V, ht) + self.c
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
            oh = [0]*self.K
            oh[idx] = 1
            Y_pred.append(oh)
        Y_pred = torch.tensor(Y_pred, dtype=torch.float).t()

        s_pred = ''
        for i in range(Y_pred.shape[1]):
            idx = torch.where(Y_pred[:, i] == 1)[0].item()
            s_pred += self.ind_to_char[idx]

        return s_pred, Y_pred, A, H, P, ht
    
    def compute_loss(self, Y, P, back = False):
        log_probs = torch.log(P)
        cross_entropy = -torch.sum(Y * log_probs)
        loss = cross_entropy.item()
        if back:
            cross_entropy.backward()
        return loss

    
    def backward(self, X, Y, A, H, P, hprev):
        dA = torch.zeros_like(A)
        dH = torch.zeros_like(H)

        G = -(Y - P)
        dV = torch.matmul(G, H.t())
        dhtau = torch.matmul(G[:, -1], self.V)
        datau = (1 - torch.pow(torch.tanh(A[:, -1]), 2)) * dhtau
        dH[:, -1] = dhtau.squeeze()
        dA[:, -1] = datau.squeeze()

        for i in range(self.seq_length - 2, -1, -1):
            dht = torch.matmul(G[:, i], self.V) + torch.matmul(dA[:, i+1].reshape(1, -1), self.W)
            dat = (1 - torch.pow(torch.tanh(A[:, i]), 2)) * dht
            dH[:, i] = dht.squeeze()
            dA[:, i] = dat.squeeze()

        Hd = torch.cat((hprev, H[:, :-1]), dim=1)
        dW = torch.matmul(dA, Hd.t())
        dU = torch.matmul(dA, X.t())
        dc = G.sum(1).reshape((-1, 1))
        db = dA.sum(1).reshape((-1, 1))
        grads = {'U': dU, 'W': dW, 'V': dV, 'c': dc, 'b': db}
        grads_clamped = {k: torch.clamp(v, -5.0, 5.0) for (k,v) in grads.items()}
        return grads, grads_clamped

    def synthetize_seq(self, h0, x0, n):
        """
        Forward pass: synthetizes a character sequence of length n.
        """
        t, ht, xt = 0, h0, x0
        indexes = []
        while t < n:
            at = torch.matmul(self.W, ht) + torch.matmul(self.U, xt) + self.b
            ht = torch.nn.functional.tanh(at)
            ot = torch.matmul(self.V, ht) + self.c
            pt = torch.nn.functional.softmax(ot, dim=0)
            cp = torch.cumsum(pt, dim=0)
            a = torch.rand(1)
            ixs = torch.where(cp - a > 0)
            ii = ixs[0][0].item()
            indexes.append(ii)
            t += 1
        Y = []
        for idx in indexes:
            oh = [0]*self.K
            oh[idx] = 1
            Y.append(oh)
        Y = torch.tensor(Y).t()
        
        s = ''
        for i in range(Y.shape[1]):
            idx = torch.where(Y[:, i] == 1)[0].item()
            s += self.ind_to_char[idx]
        
        return Y, s
    
    def encode_char(self, char):
        oh = [0]*self.K
        oh[self.char_to_ind[char]] = 1
        return oh
    
    def encode_string(self, chars):
        M = []
        for i in range(len(chars)):
            M.append(self.encode_char(chars[i]))
        M = torch.tensor(M, dtype=torch.float).t()
        return M
    
    def train_alt(self, n_epochs):
        pass
    
    def train(self, n_epochs):
        mb = torch.zeros_like(self.b, dtype=torch.float)
        mc = torch.zeros_like(self.c, dtype=torch.float)
        mU = torch.zeros_like(self.U, dtype=torch.float)
        mV = torch.zeros_like(self.V, dtype=torch.float)
        mW = torch.zeros_like(self.W, dtype=torch.float)
        m = {'b': mb, 'c': mc, 'U': mU, 'V': mV, 'W': mW}

        smooth_loss = 0
        hprev = self.h0

        for i in tqdm(range(n_epochs)):
            print(f"Epoch {i+1}/{n_epochs}")
            step = 0
            for j in tqdm(range(0, len(self.book_data) // self.seq_length, self.seq_length), desc="Processing sequences"):
                X_chars = self.book_data[j:j+self.seq_length]
                Y_chars = self.book_data[j+1:j+self.seq_length+1]
                X_train = self.encode_string(X_chars)
                Y_train = self.encode_string(Y_chars)
                _, _, A_train, H_train, P_train, ht = self.forward(X_train, Y_train, hprev)
                loss = self.compute_loss(Y_train, P_train)
                _, grads_clamped = self.backward(X_train, Y_train, A_train, H_train, P_train, hprev)
                hprev = ht

                for k in self.params.keys():
                    m[k] += grads_clamped[k]**2
                    self.params[k] -= (self.eta/(m[k] + self.epsilon))*grads_clamped[k]

                smooth_loss += 0.999*smooth_loss + 0.001*loss
                if step % 100 == 0:
                    print(f"\t * Smooth loss: {smooth_loss}")
                if step % 500 == 0:
                    _, s_syn = self.synthetize_seq(hprev, X_train[:, 0], 200)
                    print(f"\t * Synthetized sequence: {s_syn}")
                step += 1
            hprev = self.h0
            
    def run(self):
        pass
    
    def check_grads(self):
        _, _, A, H, P = self.forward(self.X, self.Y)
        grads, grads_clamped = self.backward(self.X, self.Y, A, H, P)
        self.compute_loss(self.X, self.Y, True)
        print("-------- Gradient validation --------")
        print("Max diff for gradient of b:", torch.max(torch.abs(self.b.grad - grads['b'])).item())
        print("Max diff for gradient of c:", torch.max(torch.abs(self.c.grad - grads['c'])).item())
        print("Max diff for gradient of W:", torch.max(torch.abs(self.W.grad - grads['W'])).item())
        print("Max diff for gradient of U:", torch.max(torch.abs(self.U.grad - grads['U'])).item())
        print("Max diff for gradient of V:", torch.max(torch.abs(self.V.grad - grads['V'])).item())

    
### TESTING ###
rnn = RNN()
"""#rnn.synthetize_seq(torch.normal(0.0, 0.01, (M, 1)), torch.normal(0.0, 0.1, (K, 1)), 16)
s_pred, Y_pred, A, H, P = rnn.forward(rnn.X, rnn.Y)
loss = rnn.compute_loss(rnn.X, rnn.Y)
grads = rnn.backward(rnn.X, Y_pred, A, H, P)
print(f"Predicted sequence: {s_pred}", f"Current loss: {loss}", sep='\n')
print(grads['U'].shape, grads['W'].shape, grads['V'].shape, grads['c'].shape, grads['b'].shape, sep='\n')
print(rnn.U.shape, rnn.W.shape, rnn.V.shape, rnn.c.shape, rnn.b.shape, sep='\n')"""
rnn.train(2)
###############