from utils.config import *
import dgl
from dgl.data import DGLDataset, save_graphs, load_graphs
from dgl.nn.pytorch import EdgeWeightNorm
from dgl.data.utils import save_info, load_info
import os
from tqdm import tqdm
import pickle
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
import torch
import scipy.sparse as sp

params = config()


class lncRNADataset(DGLDataset):
    """
    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(lncRNADataset, self).__init__(name='lncrna',
                                            url=url,
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)
        print('Init function is being executed...')

        print('Dataset initialization is completed!')

    def process(self):
        # Process raw data into graphs, labels
        print('Process function is being executed...')

        self.kmer = params.kmer

        # Read the lncRNA data set
        self.rawdata = []
        with open(self.raw_dir, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    l = line[1:]
                    self.rawdata.append(l.split('|'))
                else:
                    self.rawdata[-1].append(''.join([i if i in 'AUCG' else '' for i in line.replace('T', 'U')]))

        # Match labels and ids
        self.lab2id, self.id2lab = {}, []
        cnt = 0
        for i in self.rawdata:
            if i[1] not in self.lab2id:
                self.lab2id[i[1]] = cnt
                self.id2lab.append(i[1])
                cnt += 1
        self.classNum = cnt

        self.kmers2id, self.id2kmers = {"<EOS>": 0}, ["<EOS>"]
        kmersCnt = 1

        # Convert the raw sequence to k-mer form, and get the labels of the data set
        self.kmerseq, rawlab = [[i[2][j:j + self.kmer] for j in range(len(i[2]) - self.kmer + 1)] for i in
                                self.rawdata], [i[1] for i in self.rawdata]

        # Match k-mers and ids
        for rna in self.kmerseq:
            for unit in rna:
                if unit not in self.kmers2id:
                    self.kmers2id[unit] = kmersCnt
                    self.id2kmers.append(unit)
                    kmersCnt += 1
        self.kmerNum = kmersCnt

        # Convert the k-mer sequence to id form
        self.idseq = np.array([[self.kmers2id[i] for i in s] for s in self.kmerseq], dtype=object)
        # Converts the labels of the data set to the id form
        self.idlab = torch.tensor([self.lab2id[i] for i in rawlab], dtype=torch.int64)

        # Train k-mer vocabulary embedding vectors as node features
        self.vector = {}
        self.generate_nodefeas()

        # Convert each sequence in the data set to a graph
        self.graphs = []
        for item in tqdm(self.idseq):
            adj_matrix = np.zeros((self.kmerNum - 1, self.kmerNum - 1))
            for i in range(len(item) - 1):
                adj_matrix[item[i] - 1][item[i + 1] - 1] += 1
            tmp_coo = sp.coo_matrix(adj_matrix)
            edge_attr = tmp_coo.data
            edge_index = np.vstack((tmp_coo.row, tmp_coo.col))
            u, v = torch.tensor(edge_index[0]), torch.tensor(edge_index[1])
            weight = torch.FloatTensor(edge_attr)
            graph = dgl.graph((u, v), num_nodes=4 ** params.kmer)
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph, weight)
            graph.edata['weight'] = norm_weight
            node_features = self.vector['embedding'][1:]
            graph.ndata['attr'] = torch.tensor(node_features)
            graph = dgl.add_self_loop(graph, ['weight'], 1.0)
            self.graphs.append(graph)

        print("Process function is done!")

    def __getitem__(self, idx):
        # Get a sample corresponding to it by idx
        return self.graphs[idx], self.idlab[idx]

    def __len__(self):
        # Number of data samples
        return len(self.graphs)

    def save(self):
        # Save the processed data to `self.save_path`
        print('Save function is being executed...')
        if not os.path.exists(self.save_dir_1):
            os.makedirs(self.save_dir_1)
        graph_path = self.save_dir_1 + f"/{params.kmer}mer_d{params.d}_{params.method}.bin"
        info_path = self.save_dir_1 + f"/{params.kmer}mer_d{params.d}_{params.method}_info.pkl"
        save_graphs(graph_path, self.graphs, {'labels': self.idlab})
        # Save additional information in the Python dictionary
        info = {'kmers2id': self.kmers2id, 'id2kmers': self.id2kmers, 'lab2id': self.lab2id, 'id2lab': self.id2lab}
        save_info(info_path, info)
        print("Save function is done!")

    def load(self):
        # Import processed data from `self.save_path`
        print('Load function is being executed...')
        graph_path = self.save_dir_1 + f"/{params.kmer}mer_d{params.d}_{params.method}.bin"
        info_path = self.save_dir_1 + f"/{params.kmer}mer_d{params.d}_{params.method}_info.pkl"
        self.graphs, label_dict = load_graphs(graph_path)
        self.idlab = label_dict['labels']
        info = load_info(info_path)
        self.kmers2id, self.id2kmers, self.lab2id, self.id2lab = info['kmers2id'], info['id2kmers'], info['lab2id'], \
                                                                 info['id2lab']

        print(f'Load dataset from {graph_path} and {info_path}!')
        print("Load function is done!")

    def has_cache(self):
        # Check if there is processed data in `self.save_path`
        print('Has_cache function is being executed...')

        self.save_dir_1 = self.save_dir + "/" + self.raw_dir.split('/')[-1].split('.')[0]

        graph_path = self.save_dir_1 + f"/{params.kmer}mer_d{params.d}_{params.method}.bin"
        info_path = self.save_dir_1 + f"/{params.kmer}mer_d{params.d}_{params.method}_info.pkl"
        print("Has_cache function is done!")
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def generate_nodefeas(self, method=params.method, feaSize=params.d, window=5, sg=1,
                          workers=8, loadCache=True):
        # Generate node features
        print('Generate_nodefeas function is being executed...')

        saveDir = 'checkpoints/Node_features/' + self.raw_dir.split('/')[-1].split('.')[0]
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        if os.path.exists(f'{saveDir}/{method}_k{params.kmer}_d{feaSize}.pkl') and loadCache:
            with open(f'{saveDir}/{method}_k{params.kmer}_d{feaSize}.pkl', 'rb') as f:
                if method == 'kmers':
                    tmp = pickle.load(f)
                    self.vector['encoder'], self.kmersFea = tmp['encoder'], tmp['kmersFea']
                else:
                    self.vector['embedding'] = pickle.load(f)
            print(f'Load cache from {saveDir}/{method}_k{params.kmer}_d{feaSize}.pkl!')
            print('Generate_nodefeas function is done!')
            return
        if method == 'word2vec':
            doc = [i + ['<EOS>'] for i in self.kmerseq]
            model = Word2Vec(doc, min_count=0, window=window, vector_size=feaSize, workers=workers, sg=sg, seed=10)
            word2vec = np.zeros((self.kmerNum, feaSize), dtype=np.float32)
            for i in range(self.kmerNum):
                word2vec[i] = model.wv[self.id2kmers[i]]
            self.vector['embedding'] = word2vec
        elif method == 'glove':
            from glove import Glove, Corpus
            doc = [i + ['<EOS>'] for i in self.kmerseq]
            corpus = Corpus()
            corpus.fit(doc, window=window)
            glove = Glove(no_components=feaSize)
            glove.fit(corpus.matrix, epochs=100, no_threads=workers, verbose=False)
            glove.add_dictionary(corpus.dictionary)
            gloveVec = np.zeros((self.kmerNum, feaSize), dtype=np.float32)
            for i in range(self.kmerNum):
                gloveVec[i] = glove.word_vectors[glove.dictionary[self.id2kmers[i]]]
            self.vector['embedding'] = gloveVec

        elif method == 'kmers':
            enc = OneHotEncoder(categories='auto')
            enc.fit([[i] for i in self.kmers2id.values()])
            feaSize = len(self.kmers2id)
            kmers = np.zeros((len(self.labels), feaSize))
            bs = 50000
            print('Getting the kmers vector...')
            for i, t in enumerate(self.idseq):
                for j in range((len(t) + bs - 1) // bs):
                    kmers[i] += enc.transform(np.array(t[j * bs:(j + 1) * bs]).reshape(-1, 1)).toarray().sum(
                        axis=0)
            kmers = kmers[:, 1:]
            feaSize -= 1
            # Normalized
            kmers = (kmers - kmers.mean(axis=0)) / kmers.std(axis=0)
            self.vector['encoder'] = enc
            self.kmersFea = kmers
        # Save vectors
        with open(f'{saveDir}/{method}_k{params.kmer}_d{feaSize}.pkl', 'wb') as f:
            if method == 'kmers':
                pickle.dump({'encoder': self.vector['encoder'], 'kmersFea': self.kmersFea}, f, protocol=4)
            else:
                pickle.dump(self.vector['embedding'], f, protocol=4)

        print('Generate_nodefeas function is done!')
        return
