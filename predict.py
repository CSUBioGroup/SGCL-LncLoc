from models.classifier import *
import os, re
import pickle
import dgl
from dgl.nn.pytorch import EdgeWeightNorm
import scipy.sparse as sp
import numpy as np


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


device = torch.device('cpu')
kmers2id = load_variavle("checkpoints/kmers2id.pkl")
embedding = load_variavle("checkpoints/embedding.pkl")


# Convert the raw RNA sequence into a graph
def item2graph(seq, kmers2id=kmers2id, embedding=embedding, device=device):
    seq = ''.join([i if i in 'AUCG' else '' for i in seq.replace('T', 'U')])
    # This '-1' makes idseq consistent with the index of the adjacency matrix.
    idseq = [kmers2id[seq[i:i + 5]] - 1 for i in range(len(seq) - 5 + 1)]
    adj_matrix = np.zeros((len(kmers2id), len(kmers2id)))
    for i in range(len(idseq) - 1):
        adj_matrix[idseq[i]][idseq[i + 1]] += 1
    tmp_coo = sp.coo_matrix(adj_matrix)
    edge_attr = tmp_coo.data
    edge_index = np.vstack((tmp_coo.row, tmp_coo.col))
    u, v = torch.tensor(edge_index[0]), torch.tensor(edge_index[1])
    weight = torch.FloatTensor(edge_attr)
    graph = dgl.graph((u, v), num_nodes=4 ** 5)
    norm = EdgeWeightNorm(norm='both')
    norm_weight = norm(graph, weight)
    graph.edata['weight'] = norm_weight
    graph.ndata['attr'] = embedding
    graph = dgl.add_self_loop(graph, ['weight'], 1.0)
    return graph.to(device), idseq


def predict(rnaseq):
    model_path = "checkpoints/Dgl_models/final_models/"
    # Get all the file names in the directory
    file_names = os.listdir(model_path)
    # Use regular expressions to match file names and save them in a list
    matches = [[m[0], m[1]] for f in file_names if (m := re.match('cv(\d)+.pkl', f))]
    # Sort by the number in the file name
    matches.sort(key=lambda x: x[1])

    model1 = GraphLncLoc2_alpha(embedding=torch.rand(1024, 128), device=device)
    model1.load_state_dict(torch.load(f"{model_path}{matches[0][0]}", map_location=device))
    model1.eval()

    model2 = GraphLncLoc2_alpha(embedding=torch.rand(1024, 128), device=device)
    model2.load_state_dict(torch.load(f"{model_path}{matches[1][0]}", map_location=device))
    model2.eval()

    model3 = GraphLncLoc2_alpha(embedding=torch.rand(1024, 128), device=device)
    model3.load_state_dict(torch.load(f"{model_path}{matches[2][0]}", map_location=device))
    model3.eval()

    model4 = GraphLncLoc2_alpha(embedding=torch.rand(1024, 128), device=device)
    model4.load_state_dict(torch.load(f"{model_path}{matches[3][0]}", map_location=device))
    model4.eval()

    model5 = GraphLncLoc2_alpha(embedding=torch.rand(1024, 128), device=device)
    model5.load_state_dict(torch.load(f"{model_path}{matches[4][0]}", map_location=device))
    model5.eval()
    model_list = [model1, model2, model3, model4, model5]

    res = 0.0
    alpha_node = np.zeros(1024)
    graph, idseq = item2graph(rnaseq)
    for _, model in enumerate(model_list):
        x1, _, alpha = model(graph)
        res += x1.detach()
        alpha_node += alpha.squeeze(1).detach().numpy()
    res /= 5
    alpha_node /= 5
    alpha_seq = np.zeros(len(rnaseq))
    for i in range(len(idseq)):
        if i < 4:
            alpha_seq[i] = np.mean(alpha_node[idseq[:i + 1]])
        else:
            alpha_seq[i] = np.mean(alpha_node[idseq[i - 4:i + 1]])

    for j in range(-4, 0, 1):
        alpha_seq[j] = np.mean(alpha_node[idseq[j:]])

    # Normalized the attention weight
    total = np.sum(alpha_seq)
    normalized_alpha_seq = alpha_seq / total

    return res, normalized_alpha_seq


if __name__ == '__main__':
    rnaseq = "GAGAAGGGAGGAGTTATTCAGGCCTCCGCCAGCTTCTAGGCCCTGGGGATGGTCTTTCACCTCCCTCTTTCTGATCTCTTTTTCATGCTCCTCCTTGCTCCAAAGAAAAGCCGGATGGCAAAAGAGCCCAGAACCTATTGGAACTGACAAAATCAAGTCACGGCGCCTACAAAGATGAGGGGCAGATTCTGGCTGCCTTTTAATTTCGTCCTTCACCTGATATCTGTGCCAGAGAATGATAAAAATCATAATAAAGGAAATAATGGAAGAGGAGACTTATGTTACTGGGGACATCTAACATAATTATTTTCCTGATTCAGTGGCATGGTTCAGTCTTCCAGGAGTTCTGCTACAGAGAAGAGAGTAACCCCCATCCATCATGGCCAAAGCACCCAGTCAGGCTCCGCTCTGGATCCAGCCCGACAAATGCAACCCTTGAATAGGGTTTGTGCAAGCAAACTGGATGACGACCGAAGAAACCCTGTCGCTTCTGAGAAGACACCCAATCCAAGAATGTGAGTTCTGGAAATGTCATTAAATGTCAGTTATATACATGCAAAAAAAAAAAAAAAAA"
    prob, alpha_seq = predict(rnaseq)
    print(prob, alpha_seq.shape)
