# SGCL-LncLoc
An interpretable deep learning model for improving lncRNA subcellular localization prediction with supervised graph contrastive learning
# Requirements
    dgl==0.9.1
    gensim==4.2.0
    glove_python_binary==0.2.0
    numpy==1.23.1
    optuna==3.1.0
    python==3.8.13
    scikit_learn==1.1.1
    scipy==1.9.1
    torch==1.12.1
    tqdm==4.64.1 

# Usage
## Simple usage
You can train the model in a very simple way by the command blow:
``python train.py >output/log.txt`` 
## How to train your own model
Also you can use the package provided by us to train your model.

First, you need to import the package.  
```python
from data.lncRNADataset import *
from cv_train import *
```
Second, you can train your own model by modifying the variables in *utils/config.py*. 

>In the *utils/config.py*, the meaning of the variables is explained as follows:
>>***k*** is the value of the k-mer nodes.  
>>***d*** is the dimension of vector of node features.  
>>***method*** is the encoding method of node features.  
>>***hidden_dim*** is the number of the hidden neurons of GCNs.  
>>***alpha*** is the weight factor of the supervised contrastive learning loss function.  
>>***savePath*** is the folder where the model is saved.  
>>***device*** is the device you used to build and train the model. It can be "cpu" for cpu or "cuda" for gpu, and "cuda:0" for gpu 0.  

Then you need to create the data object.  
```python
dataset = lncRNADataset(raw_dir="data/dataset.txt", save_dir=f'checkpoints/Dgl_graphs')
```
Finally, you can create the training object for models and start training.
```python
cv_models = cv_train()
cv_models.train(dataset)
```

## How to do prediction
First, import the package. 
```python
from predict import *
```
Then assume an RNA sequence.
```python
rnaseq = "GAGAAGGGAGGAGTTATTCAGGCCTCCGCCAGCTTCTAGGCCCTGGGGATGGTCTTTCACCTCCCTCTTTCTGATCTCTTTTTCATGCTCCTCCTTGCTCCAAAGAAAAGCCGGATGGCAAAAGAGCCCAGAACCTATTGGAACTGACAAAATCAAGTCACGGCGCCTACAAAGATGAGGGGCAGATTCTGGCTGCCTTTTAATTTCGTCCTTCACCTGATATCTGTGCCAGAGAATGATAAAAATCATAATAAAGGAAATAATGGAAGAGGAGACTTATGTTACTGGGGACATCTAACATAATTATTTTCCTGATTCAGTGGCATGGTTCAGTCTTCCAGGAGTTCTGCTACAGAGAAGAGAGTAACCCCCATCCATCATGGCCAAAGCACCCAGTCAGGCTCCGCTCTGGATCCAGCCCGACAAATGCAACCCTTGAATAGGGTTTGTGCAAGCAAACTGGATGACGACCGAAGAAACCCTGTCGCTTCTGAGAAGACACCCAATCCAAGAATGTGAGTTCTGGAAATGTCATTAAATGTCAGTTATATACATGCAAAAAAAAAAAAAAAAA"
```
Finally, we call the function `` predict``   to obtain the prediction probability and the attention weight that varies along the sequence.
```python
prob, alpha_seq = predict(rnaseq)
print(prob, alpha_seq)
```
Note: The final output value is the predicted probability that the sequence is located in the nucleus.

## Independent test set
The *testset.txt* in *Independent_test_set* folder is used in comparison with other predictors. 

## Other details
The other details can be seen in the paper and the codes.

# Citation
Min Li, Baoying Zhao, Yiming Li, Jingwei Lu, Fuhao Zhang, Shichao Kan, Min Zeng. SGCL-LncLoc: an interpretable deep learning model for improving lncRNA subcellular localization prediction with supervised graph contrastive learning.

# License
This project is licensed under the MIT License - see the LICENSE.txt file for details
