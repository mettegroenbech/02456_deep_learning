import torch
import torch.nn as nn
import torch.nn.functional as f

class BagOfWords_SST(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(BagOfWords_SST, self).__init__()

        embedding_dim = TEXT.vocab.vectors.size()[1]
        num_embeddings = TEXT.vocab.vectors.size()[0]
        num_classes = len(LABEL.vocab.itos)

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        
        self.translate = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)

        self.l_1 = nn.Linear(in_features=embedding_dim,
                          out_features=30,
                          bias=True)
        self.l_2 = nn.Linear(in_features=30,
                          out_features=30,
                          bias=True)
        self.l_out = nn.Linear(in_features=30,
                            out_features=num_classes,
                            bias=False)

        self.dropout = nn.Dropout(p=0.35)
        self.activation = nn.ReLU()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
    def forward(self, x):

        x = self.embeddings(x)
        x = self.activation(self.translate(x))
        x = self.dropout(torch.mean(x, dim=0))
        
        x = self.activation(self.l_1(x))
        x = self.dropout(x)
        x = self.activation(self.l_2(x))
        x = self.dropout(x)

        out = f.softmax(self.l_out(x), dim=1)
        return {'out':out}

class BagOfWords_SNLI(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(BagOfWords_SNLI, self).__init__()

        embedding_dim = TEXT.vocab.vectors.size()[1]
        num_embeddings = TEXT.vocab.vectors.size()[0]
        num_classes = len(LABEL.vocab.itos)

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)
        
        self.translate = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)

        self.l_1 = nn.Linear(in_features=embedding_dim*2,
                          out_features=300,
                          bias=True)
        self.l_2 = nn.Linear(in_features=300,
                          out_features=300,
                          bias=True)
        self.l_out = nn.Linear(in_features=300,
                            out_features=num_classes,
                            bias=False)

        self.dropout = nn.Dropout(p=0.35)
        self.activation = nn.ReLU()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, premise, hypothesis):
        
        p = self.embeddings(premise)
        h = self.embeddings(hypothesis)
        
        p = self.activation(self.translate(p))
        h = self.activation(self.translate(h))

        p = torch.mean(p, dim=0)
        h = torch.mean(h, dim=0)
        x = self.dropout(torch.cat((p, h), 1))

        x = self.activation(self.l_1(x))
        x = self.dropout(x)
        x = self.activation(self.l_2(x))
        x = self.dropout(x)

        out = f.softmax(self.l_out(x), dim=1)
        return {'out':out}

class LSTM_RNN_SST(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(LSTM_RNN_SST, self).__init__()

        NUM_LSTM_LAYERS = 1
        HIDDEN_SIZE = 100 

        embedding_dim = TEXT.vocab.vectors.size()[1]
        num_embeddings = TEXT.vocab.vectors.size()[0]
        num_classes = len(LABEL.vocab.itos)

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)

        self.embeddings.weight.detach_()

        self.translate = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True) 
        
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LSTM_LAYERS,
                            bidirectional=True)

        self.l_1 = nn.Linear(in_features=4*HIDDEN_SIZE,
                          out_features=2*HIDDEN_SIZE,
                          bias=True)
        self.l_2 = nn.Linear(in_features=2*HIDDEN_SIZE,
                          out_features=HIDDEN_SIZE,
                          bias=True)
        self.l_3 = nn.Linear(in_features=HIDDEN_SIZE,
                          out_features=HIDDEN_SIZE,
                          bias=True)

        self.l_out = nn.Linear(in_features=HIDDEN_SIZE,
                            out_features=num_classes,
                            bias=False)

        self.dropout = nn.Dropout(p=0.35)        
        self.activation = nn.ReLU()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
    def forward(self, x):

        x = self.embeddings(x)
        x = self.activation(self.translate(x))

        x, (_, _) = self.lstm(x)

        x = torch.cat((torch.mean(x, dim=0), torch.max(x, dim=0)[0]), dim=1)
        
        x = self.activation(self.l_1(x))
        x = self.dropout(x)
        x = self.activation(self.l_2(x))
        x = self.dropout(x)
        x = self.activation(self.l_3(x))
        x = self.dropout(x)

        out = f.softmax(self.l_out(x), dim=1)
        return {'out':out}

class LSTM_RNN_SNLI(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(LSTM_RNN_SNLI, self).__init__()

        NUM_LSTM_LAYERS = 1
        HIDDEN_SIZE = 200

        embedding_dim = TEXT.vocab.vectors.size()[1]
        num_embeddings = TEXT.vocab.vectors.size()[0]
        num_classes = len(LABEL.vocab.itos)

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)

        self.embeddings.weight.detach_()

        self.translate = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)      
        
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LSTM_LAYERS,
                            bidirectional=False)
        self.l_1 = nn.Linear(in_features=4*HIDDEN_SIZE,
                          out_features=2*HIDDEN_SIZE,
                          bias=True)
        self.l_2 = nn.Linear(in_features=2*HIDDEN_SIZE,
                          out_features=HIDDEN_SIZE,
                          bias=True)
        self.l_3 = nn.Linear(in_features=HIDDEN_SIZE,
                          out_features=HIDDEN_SIZE,
                          bias=True)
        self.l_out = nn.Linear(in_features=HIDDEN_SIZE,
                            out_features=num_classes,
                            bias=False)

        self.dropout = nn.Dropout(p=0.35)
        self.activation = nn.ReLU()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, premise, hypothesis):

        p = self.embeddings(premise)
        h = self.embeddings(hypothesis)

        p = self.activation(self.translate(p))
        h = self.activation(self.translate(h))

        p, (_, _) = self.lstm(p)
        h, (_, _) = self.lstm(h)

        p = torch.cat((torch.mean(p, dim=0), torch.max(p, dim=0)[0]), dim=1)
        h = torch.cat((torch.mean(h, dim=0), torch.max(h, dim=0)[0]), dim=1)
        x = torch.cat((p, h), dim=1)

        x = self.activation(self.l_1(x))
        x = self.dropout(x)
        x = self.activation(self.l_2(x))
        x = self.dropout(x)
        x = self.activation(self.l_3(x))
        x = self.dropout(x)

        out = f.softmax(self.l_out(x), dim=1)
        return {'out':out}

class BCN_SST(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(BCN_SST, self).__init__()
        
        embedding_dim = TEXT.vocab.vectors.size()[1]
        num_embeddings = TEXT.vocab.vectors.size()[0]
        num_classes = len(LABEL.vocab.itos) 

        NUM_LSTM_LAYERS = 1
        HIDDEN_SIZE = 50

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)

        self.embeddings.weight.detach_()

        self.translate = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        self.relu = nn.ReLU()
        self.translate_dropout = nn.Dropout(0.2)

        self.encoder = nn.LSTM(input_size=embedding_dim,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LSTM_LAYERS,
                            bidirectional=True)

        self.integrator = nn.LSTM(input_size=6*HIDDEN_SIZE,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LSTM_LAYERS,
                            bidirectional=True)

        self.l_1 = Maxout(in_features=16*HIDDEN_SIZE,
                          out_features=4*HIDDEN_SIZE,
                          pool_size=1)
        self.l_1_dropout = nn.Dropout(0.4)
        self.l_1_batchnorm = nn.BatchNorm1d(num_features=4*HIDDEN_SIZE)

        self.l_2 = Maxout(in_features=4*HIDDEN_SIZE,
                          out_features=num_classes,
                          pool_size=1)

        self.self_pooling_linear_1 = nn.Linear(in_features=2*HIDDEN_SIZE,
                                               out_features=1,
                                               bias=True)
        
        self.self_pooling_linear_2 = nn.Linear(in_features=2*HIDDEN_SIZE,
                                               out_features=1,
                                               bias=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001)

    def forward(self, x):
        y = x

        x = self.embeddings(x)
        y = self.embeddings(y)

        x = self.relu(self.translate(x))
        x = self.translate_dropout(x)
        y = self.relu(self.translate(y))
        y = self.translate_dropout(y)
        
        x, (_, _) = self.encoder(x)
        y, (_, _) = self.encoder(y)

        X = x.permute(1,2,0)
        Y = y.permute(1,2,0)

        A = X.bmm(Y.permute(0,2,1))
        A_x = f.softmax(A, dim=2)
        A_y = f.softmax(A.permute(0,2,1), dim=2)

        C_x = A_x.permute(0,2,1).bmm(X)
        C_y = A_y.permute(0,2,1).bmm(Y)

        tmp_x = torch.cat((X, X - C_y, X * C_y), dim=1)
        tmp_y = torch.cat((Y, Y - C_x, Y * C_x), dim=1)
        
        X_y, (_, _) = self.integrator(tmp_x.permute(2,0,1))
        Y_x, (_, _) = self.integrator(tmp_y.permute(2,0,1))
        
        beta_x = f.softmax(self.self_pooling_linear_1(X_y), dim=0)
        beta_y = f.softmax(self.self_pooling_linear_2(Y_x), dim=0)

        x_self = X_y.permute(1,2,0).bmm(beta_x.permute(1,0,2)).squeeze(2)
        y_self = Y_x.permute(1,2,0).bmm(beta_y.permute(1,0,2)).squeeze(2)

        X_y_max = torch.max(X_y, dim=0)[0]
        Y_x_max = torch.max(Y_x, dim=0)[0]

        X_y_min = torch.min(X_y, dim=0)[0]
        Y_x_min = torch.min(Y_x, dim=0)[0]

        X_y_mean = torch.mean(X_y, dim=0, keepdim=True)[0]
        Y_x_mean = torch.mean(Y_x, dim=0, keepdim=True)[0]

        X_pool = torch.cat((X_y_max, X_y_min, X_y_mean, x_self), dim=1)
        Y_pool = torch.cat((Y_x_max, Y_x_min, Y_x_mean, y_self), dim=1)

        joined = torch.cat((X_pool, Y_pool), dim=1)

        out = self.l_1(joined)
        out = self.l_1_dropout(out)
        out = self.l_1_batchnorm(out)
        out = self.l_2(out)
        out = f.softmax(out, dim=1)

        return {'out':out}

class BCN_SNLI(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(BCN_SNLI, self).__init__()
        
        embedding_dim = TEXT.vocab.vectors.size()[1]
        num_embeddings = TEXT.vocab.vectors.size()[0]
        num_classes = len(LABEL.vocab.itos)

        NUM_LSTM_LAYERS = 1
        HIDDEN_SIZE = 50

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.copy_(TEXT.vocab.vectors)

        self.translate = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        self.relu = nn.ReLU()
        self.translate_dropout = nn.Dropout(0.2)

        self.encoder = nn.LSTM(input_size=embedding_dim,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LSTM_LAYERS,
                            bidirectional=True)

        self.integrator = nn.LSTM(input_size=6*HIDDEN_SIZE,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LSTM_LAYERS,
                            bidirectional=True)

        self.l_1 = Maxout(in_features=16*HIDDEN_SIZE,
                          out_features=4*HIDDEN_SIZE,
                          pool_size=1)
        self.l_1_dropout = nn.Dropout(0.4)
        self.l_1_batchnorm = nn.BatchNorm1d(num_features=4*HIDDEN_SIZE)

        self.l_2 = Maxout(in_features=4*HIDDEN_SIZE,
                          out_features=num_classes,
                          pool_size=1)

        self.self_pooling_linear = nn.Linear(in_features=2*HIDDEN_SIZE,
                                               out_features=1,
                                               bias=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, premise, hypothesis):
        x, y = premise, hypothesis
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensuring that the premise and hypothesis have the same size
        if x.size() > y.size():
            tmp = torch.ones(x.size(), dtype=torch.long, device=DEVICE)
            tmp[:y.size()[0],:] = y
            y = tmp
        elif x.size() < y.size():
            tmp = torch.ones(y.size(), dtype=torch.long, device=DEVICE)
            tmp[:x.size()[0],:] = x
            x = tmp

        x = self.embeddings(x)
        y = self.embeddings(y)

        x = self.relu(self.translate(x))
        x = self.translate_dropout(x)
        y = self.relu(self.translate(y))
        y = self.translate_dropout(y)
        
        x, (_, _) = self.encoder(x)
        y, (_, _) = self.encoder(y)

        X = x.permute(1,2,0)
        Y = y.permute(1,2,0)
        
        A = X.bmm(Y.permute(0,2,1))
        A_x = f.softmax(A, dim=2)
        A_y = f.softmax(A.permute(0,2,1), dim=2)

        C_x = A_x.permute(0,2,1).bmm(X)
        C_y = A_y.permute(0,2,1).bmm(Y)

        tmp_x = torch.cat((X, X - C_y, X * C_y), dim=1)
        tmp_y = torch.cat((Y, Y - C_x, Y * C_x), dim=1)
        
        X_y, (_, _) = self.integrator(tmp_x.permute(2,0,1))
        Y_x, (_, _) = self.integrator(tmp_y.permute(2,0,1))
        
        beta_x = f.softmax(self.self_pooling_linear(X_y), dim=0)
        beta_y = f.softmax(self.self_pooling_linear(Y_x), dim=0)

        x_self = X_y.permute(1,2,0).bmm(beta_x.permute(1,0,2)).squeeze(2)
        y_self = Y_x.permute(1,2,0).bmm(beta_y.permute(1,0,2)).squeeze(2)

        X_y_max = torch.max(X_y, dim=0)[0]
        Y_x_max = torch.max(Y_x, dim=0)[0]

        X_y_min = torch.min(X_y, dim=0)[0]
        Y_x_min = torch.min(Y_x, dim=0)[0]

        X_y_mean = torch.mean(X_y, dim=0, keepdim=True)[0]
        Y_x_mean = torch.mean(Y_x, dim=0, keepdim=True)[0]

        X_pool = torch.cat((X_y_max, X_y_min, X_y_mean, x_self), dim=1)
        Y_pool = torch.cat((Y_x_max, Y_x_min, Y_x_mean, y_self), dim=1)

        joined = torch.cat((X_pool, Y_pool), dim=1)

        out = self.l_1(joined)
        out = self.l_1_dropout(out)
        out = self.l_1_batchnorm(out)
        out = self.l_2(out)
        out = f.softmax(out, dim=1)

        return {'out':out}

class Maxout(nn.Module): # Taken from https://github.com/pytorch/pytorch/issues/805
    def __init__(self, in_features, out_features, pool_size):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = in_features, out_features, pool_size
        self.lin = nn.Linear(in_features, out_features * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m