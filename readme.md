How to run the code:
=======
`python reranker.py train DEBUG B`

where  
`train` is training and save model first, if using other word, then just load models without training.   
`DEBUG` use small dataset. If using other word or None, use standard data.   
`B` means LSTM with CNN model.   


Data format
=======
Generate the nbest results using neural BI-LSTM-CRF at [NbestNER](https://github.com/jiesutd/LasagneNLP).
Generate the nbest results using discrete CRF model at [CRF++](https://taku910.github.io/crfpp/)
Data format follow the [sample data](/data/)



Folder
=======
data: where the data saved
model: model file saved
results: training model saved
utils: load data, metric file saved


Cite: 
========
    @article{yang2017neural,  
     title={Neural Reranking for Named Entity Recognition},  
     author={Yang, Jie and Zhang, Yue and Dong, Fei},  
     booktitle = {Proceedings of RANLP},
     year={2017}  
    }  