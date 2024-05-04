import torch
import torch.nn.functional as F
from rnnModel.data.make_dataset import make_charmap

class RNN:
    def __init__(self, m = 100, eta = 0.1, epsilon = 1e-8, seq_length = 25, data_path = "./data/goblet_book.txt", autograd = False) -> None:
        ## Constants
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m = m  # dimensionality of the hidden state
        self.eta = eta  # learning rate
        self.epsilon = epsilon  # for AdaGrad
        self.seq_length = seq_length  # length of input sequences used during training
        self.book_data, self.K, self.char_to_ind, self.ind_to_char = make_charmap(data_path)  # dimensionality of input and character mappings

        ## Parameters
        # bias vectors
        self.b = torch.zeros((self.m, 1), dtype=torch.float, device=self.device)
        self.c = torch.zeros((self.K, 1), dtype=torch.float, device=self.device)
        # weight matrices
        sig = 0.01
        self.U = torch.normal(0.0, sig, (self.m, self.K), dtype=torch.float, device=self.device)
        self.W = torch.normal(0.0, sig, (self.m, self.m), dtype=torch.float, device=self.device)
        self.V = torch.normal(0.0, sig, (self.K, self.m), dtype=torch.float, device=self.device)
        self.h0 = torch.zeros((self.m, 1), dtype=torch.float, device=self.device)
        self.params = {
            'W': self.W, 
            'U': self.U,
            'V': self.V,
            'b': self.b,
            'c': self.c
        }

    def forward(self, X, Y, hprev):
        ht = hprev.clone()
        indexes = []
        P = torch.zeros((self.K, self.seq_length), dtype=torch.float, device=self.device)
        A = torch.zeros((self.m, self.seq_length), dtype=torch.float, device=self.device)
        H = torch.zeros((self.m, self.seq_length), dtype=torch.float, device=self.device)
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
            a = torch.rand(1, device=self.device)
            ixs = torch.where(cp - a > 0)
            ii = ixs[0][0].item()
            indexes.append(ii)

        Y_pred = []
        for idx in indexes:
            oh = [0]*self.K
            oh[idx] = 1
            Y_pred.append(oh)
        Y_pred = torch.tensor(Y_pred, dtype=torch.float, device=self.device).t()

        s_pred = ''
        for i in range(Y_pred.shape[1]):
            idx = torch.where(Y_pred[:, i] == 1)[0].item()
            s_pred += self.ind_to_char[idx]

        log_probs = torch.log(P)
        cross_entropy = -torch.sum(Y * log_probs)

        loss = cross_entropy.item()

        return s_pred, Y_pred, A, H, P, ht, loss

    
    def backward(self, X, Y, A, H, P, hprev):
        dA = torch.zeros_like(A, device=self.device)
        dH = torch.zeros_like(H, device=self.device)

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
            xt = xt.reshape((self.K, 1))
            at = torch.matmul(self.W, ht) + torch.matmul(self.U, xt) + self.b
            ht = torch.nn.functional.tanh(at)
            ot = torch.matmul(self.V, ht) + self.c
            pt = torch.nn.functional.softmax(ot, dim=0)
            cp = torch.cumsum(pt, dim=0)
            a = torch.rand(1, device=self.device)
            ixs = torch.where(cp - a > 0)
            ii = ixs[0][0].item()
            indexes.append(ii)
            xt = torch.zeros((self.K, 1), dtype=torch.float, device=self.device)
            xt[ii, 0] = 1
            t += 1
        Y = []
        for idx in indexes:
            oh = [0]*self.K
            oh[idx] = 1
            Y.append(oh)
        Y = torch.tensor(Y, device=self.device).t()
        
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
        M = torch.tensor(M, dtype=torch.float, device=self.device).t()
        return M
    
    def train(self, n_epochs):
        e, step, epoch = 0, 0, 0
        smooth_loss = 0
        losses = []
        hprev = self.h0

        mb = torch.zeros_like(self.b, dtype=torch.float, device=self.device)
        mc = torch.zeros_like(self.c, dtype=torch.float, device=self.device)
        mU = torch.zeros_like(self.U, dtype=torch.float, device=self.device)
        mV = torch.zeros_like(self.V, dtype=torch.float, device=self.device)
        mW = torch.zeros_like(self.W, dtype=torch.float, device=self.device)
        ms = {'b': mb, 'c': mc, 'U': mU, 'V': mV, 'W': mW}

        while epoch < n_epochs:
            X_chars = self.book_data[e:e+self.seq_length]
            Y_chars = self.book_data[e+1:e+self.seq_length+1]
            X_train = self.encode_string(X_chars)
            Y_train = self.encode_string(Y_chars)

            _, _, A_train, H_train, P_train, ht, loss = self.forward(X_train, Y_train, hprev)
            grads, grads_clamped = self.backward(X_train, Y_train, A_train, H_train, P_train, hprev)

            for k in ms.keys():
                ms[k] += grads_clamped[k]**2
                self.params[k] -= (self.eta/torch.sqrt(ms[k] + self.epsilon))*grads_clamped[k]

            if step == 0:
                smooth_loss = loss
            else:
                smooth_loss = 0.999*smooth_loss + 0.001*loss

            losses.append(smooth_loss)

            if step % 1000 == 0:
                print(f"Step: {step}")
                print(f"\t * Smooth loss: {smooth_loss}")
            if step % 5000 == 0:
                _, s_syn = self.synthetize_seq(hprev, X_train[:, 0], 200)
                print("-" * 100)
                print(f"Synthetized sequence: \n{s_syn}")
                print("-" * 100)
            if step % 100000 == 0 and step > 0:
                _, s_lsyn = self.synthetize_seq(hprev, X_train[:, 0], 1000)
                print("-" * 100)
                print(f"Long synthetized sequence: \n{s_lsyn}")
                print("-" * 100)

            step += 1
            e += self.seq_length
            if e > len(self.book_data) - self.seq_length:
                e = 0
                epoch += 1
                hprev = RNN['h0']
            else:
                hprev = ht
        return losses
            
    def run(self):
        pass

    
### TESTING ###
rnn = RNN()
losses = rnn.train(2)
###############