from data.lncRNADataset import *
from cv_train import *

dataset = lncRNADataset(raw_dir="data/dataset.txt", save_dir=f'checkpoints/Dgl_graphs')
cv_models = cv_train()
cv_models.train(dataset)
