import torch


# Set global variables and modify them here
class config:
    def __init__(self):
        # K value of the k-mer node
        self.kmer = 5
        # The encoding method of node features
        encoding = ['word2vec', 'glove']
        self.method = encoding[0]
        # Dimension of the node features
        self.d = 128
        # Number of the hidden neurons
        self.hidden_dim = 256
        # Seed for model initialization
        self.seed = 10
        # Number of sample categories
        self.n_classes = 2

        # Training parameters
        self.batchSize = 16
        self.numEpochs = 1000
        self.lr = 0.0005
        self.earlyStop = 100
        self.kFold = 5
        # The weight factor of the supervised contrastive learning loss function
        self.alpha = 0.1
        # Models saved here
        self.savePath = f"checkpoints/Dgl_models/k{self.kmer}_d{self.d}_h{self.hidden_dim}_lr{self.lr}_b{self.batchSize}_{self.method[0]}_a{self.alpha}_test/"

        self.device = torch.device("cuda:0")
