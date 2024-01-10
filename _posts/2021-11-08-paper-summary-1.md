---
layout: post
title: Paper Summary 1
date: 2021-11-08 16:40:16
description: paper summary
tags: formatting links
categories: sample-posts
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>


- **Paper**:<a name='1'></a>

###                                **Label-Specific Dual Graph Neural Network for Multi-Label Text Classification**   [[paper]](https://aclanthology.org/2021.acl-long.298/) 

- **Background and motivation**:

​       Label-specific semantic information is ignored to help distinguish similar classes when doing multi-label text classification

- **Solution**:

​       Used two GCNs, **one** encodes the information using the representation from the previous layer (which is an attention mechanism 
​       with randomly initialized label information as part of the attention score over the outputs from a BiLSTM), the adjacency matrix is a 
​       statistical matrix whose element A<sub>ij</sub> is the conditional probability of a sample belonging to class<sub>i</sub> while it belongs to class<sub>j</sub> .

​       The **other** takes the output of the previous GCN and then construct the adjacency matrix dynamically

​       Explanations are:
​       (1). the first GCN encoded label information with the training sentences.
​       (2). however the adjacency matrix for the first GCN is a statistical co-occurrence matrix, which may result in long-tail distribution
​              problem, thus the second GCN dynamically construct its adjacency matrix to capture interactive relations between components.

​       **Highlights**

​             Personally I think the shining point of this work lies in its second GCN **where the adjacency matrix is dynamically**
​       **constructed**, which in some kind of sense enables the GCN to **capture hidden interactions** and thus better help the features from
​       different nodes to merge.

- **Architecture**:

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic1.png?token=AOW4OJW5XAOAWIR77BROPNTBQ7EBI" style="zoom:70%;" /></div>



- **Paper**

###                                **Concept-Based Label Embedding via Dynamic Routing for Hierarchical Text Classification**   [[paper]](https://aclanthology.org/2021.acl-long.388.pdf)

- **Background and motivation**

​      The concepts can be used to classify different sub-labels within a same class in Hierarchical Text Classification and previous works 
​      focus on how to learn better text representations or simply use label information however ignored 'concept'. (in the following graph 
​      they're 'design' and 'distributed', **which are not included in the origin hierarchy**)

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic2.png?token=AOW4OJQZU2WVPH55XUX6BF3BQ7EBS" /></div>


- **Solution**

​       Text is encoded and passed to stacked blocks called CCM (Concept-based Classifier Module), and a **CCM**  module is consisted of
​       a EK Encoder (to encode external knowledge), **predicted soft label embedding** from the previous CCM (which will be used to help
​       classification at current hierarchy), and a **CSM** module --- where the concept is shared via Dynamic Routing (whose pseudo code is
​       very similar to Capsule Network). And at each CCM module, predictions for current hierarchy is outputted. 

​       **Highlights**

​             I think the good part of this work is its idea of **using stacked modules**, which is a common architecture in popular models like 
​       Transformer or ResNet, and the interactions from the previous module to the next one may enable the model **to extract different**
​       **features at different depth** (which is some kind of sense naturally suits the characteristics of text classification with a hierarchical
​       structure)

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic3.png?token=AOW4OJTZFW2JHH5G5BZ2GTTBQ7ECA" style="zoom:90%;" /></div>




- **Paper**

###                                **Explicit Interaction Model towards Text Classification**   [[paper]](https://arxiv.org/abs/1811.09386#) <a name='2'></a>

- **Background and motivation**

​       The interaction between words and classes is ignored, and the motivation is to explicitly model this kind of interaction

- **Solution**

​       A transformation is performed on word-level embeddings where a matrix is acted as the role of label information

​       **Highlights**

​             The only shining point might only be **its motivation of introducing interactions between classes and texts at word level**, 
​       however the way of constructing this kind of interaction is rather casual --- simply doing a transformation with the encoded words
​       with a randomly initialized matrix, without introducing some statistical information or some similarity measurements (e.g. cos 
​       similarity) while maintaining its dynamic (e.g. in [Label-Specific Dual Graph Neural Network for Multi-Label Text Classification](#1))

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic4.png?token=AOW4OJTAFQ2OAVGACNQ4IE3BQ7ECU" /></div>



- **Paper**

###                                **Joint Embedding of  Words and Labels  for Text  Classification**  [[paper]](https://aclanthology.org/P18-1216.pdf) <a name='3'></a>

- **Background and motivation**

​       Embeds both the text and label in the same word embedding space and carries out an attention operation over and text and label to 
​       attend to relevant words.

- **Solution**

​       By embedding texts and labels in the same space, a cos similarity matrix is calculated between the text and label, the matrix is then 
​       converted to attention score to aggregate the text words to a universal representation used for classification.

​      **Highlights**

​            The idea and model are simple, however it shows good structure as well as more convincing way of leveraging label information 
​      (**used cos similarity** unlike in [Explicit Interaction Model towards Text Classification](#2) where label representation is randomly 
​       initialized)

- **Architecture**


<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic5.png?token=AOW4OJSMJ2C5LHQPTMXHP5LBQ7EDG" /></div>




- **Paper**

###                          **Multi-Task Label Embedding for Text Classification**   [[paper]](https://arxiv.org/abs/1710.07210#)


- **Background and motivation**

​       Previous works ignored label information thus result in semantic label information loss, this work leverages semantic label 
​       information and shows promising performance on task transferring.

- **Solution**

​       Texts and respective labels are grouped at task level and then embedded, for the **Input Encoder**, words in the text are first 
​       converted to word embeddings (**embedding layer**) then feed into a BiLSTM, the concatenation of the outputs from the BiLSTM at 
​       last time step is treated as the final representation for the texts (**learning layer**). For the **Label Encoder**, words in the label are first 
​       converted to word embeddings (**embedding layer**) then their average is treated as the final representation for labels (**learning
​       layer**). Matching probability is calculated using a MLP for each task, the training loss is Cross Entropy with weights for each task. 

​       For task transfer, there are three ways of updating the tasks --- Hot Update, Cold Update, Zero Update

​       **Hot Update**: when a task C **with label annotations** is added, using **only newly added task C** to keep training the trained model

​       **Cold Update**: when a task C **with label annotations** is added, using **all the available tasks** (previous+task C) to train the model 
​                              from start.

​        **Zero Update**: when a task D **without label annotation** is added, applying the newly added task D on the already trained model to 
​                               see its transfer ability.

​        Note: task **with** label annotation means there are {text, label} pairs you know corresponding label for a text (**able to train**), 
​                  **without** means there isn't such pair (**unable to train, only able to test**)

​        **Highlights**

​                The actually structure of the model is not very fancy (e.g.: the Matcher is simply a MLP) however the core idea of this work is 
​        amazing, which re-thinks training at task level, with label information and weighted loss (w.r.t tasks), the way how it does task 
​        transfer (Hot Update, Cold Update, Zero Update) is inspiring as well 

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic6.png?token=AOW4OJXM5MY6H573N2FV7WTBQ7EDU" /></div>




- **Paper**

###                                **Distinct Label Representations for Few-Shot Text Classification**    [[paper]](https://aclanthology.org/2021.acl-short.105.pdf)

- **Background and motivation**

​       Instances from semantically close classes makes it hard to do few-shot classification under meta learning settings, thus label 
​       information is introduced to help classification

- **Solution**

​       A multi-head attention is performed over the embedded texts and labels to attend to different features in texts (Difference Extractor), 
​       the outputs is later used for classification and mutual information between labels is calculated and minimized as an extra loss.

​       **Highlights**

​            The introduction of multi-head is good, and I reckon the mutual information loss is quite interesting (though I did not understand it 
​       at all:disappointed:, however I hope to dig into it when I have some time)

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic7.png?token=AOW4OJSZXFWIVKX7PU42UKLBQ7EEC" style="zoom:75%;" /></div>


​      

- **Paper**

###                                **Don’t Miss the Labels: Label-semantic Augmented Meta-Learner for Few-Shot Text Classification**   [[paper]](https://aclanthology.org/2021.findings-acl.245.pdf)

- **Background and motivation**

​       Previous work mainly focus on building a meta-learner using only information from texts, thus the information from label is wasted, 
​       to solve this drawback, this work boosts meta-learner with label semantics

- **Solution**

​       When training, the corresponding label is concatenated after the sentence and the next sentence prediction task of BERT is used to 
​       try to mix label information into the text representation, and the support representation can be computed in the following ways: **1.** 
​       [CLS];  **2.** average words' embeddings in the sentence;  **3.** average labels' embeddings in the sentence; 
​       For query, there are two inputting options:  **1.** origin text;  **2.** origin text concatenates all the labels;  and for output representations 
​       for queries used for classification, there are three options: **1.** [CLS] from the origin text;  **2.** [CLS] from the origin text concatenated 
​       with all the labels;  **3.** all label embeddings from the origin text concatenated with all labels;

​       **Highlights**

​               The gains may all comes from BERT instead of its proposed modifications, if encoded changed, its performance may not be 
​       guaranteed.

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic8.png?token=AOW4OJTBG357IBCCVK45V7DBQ7EEW" style="zoom:65%;" /></div>



- **Paper**

###                                **Learning to Bridge Metric Spaces: Few-shot Joint Learning of Intent Detection and Slot Filling**    [[paper]](https://arxiv.org/abs/2106.07343)

- **Background and motivation**

​       The authors consider learning intent detection and slot filling together can benefit both tasks, thus propose to learn both tasks 
​       simultaneously using prototype network under a few shot learning setting, contrastive learning is used as well to help learn better 
​       representations of the prototypes.

- **Solution**

​       Since there are two tasks training together, the prototype is merged for later classification, in detail: **1.** a cross-attention score is
​       estimated using additive attention with raw prototypes (the average of the encoded sentences) from intent detection and slot filling, 
​       **2.** and fused prototypes for two tasks are calculated using the cross-attention score over the raw prototypes, **3.** the fused and raw 
​       prototypes of each task are then merged with weight of { $\alpha$ , 1 -  $\alpha$ }, which will later be the final prototypes for classification of intent 
​       detection/slot filling.
​       For Contrastive Learning, the loss is composed of two parts, one is **Intra Contrastive Loss** (where the loss is modified with margin 
​       and calculated task-wise, the losses of two tasks are averaged as the final Intra Contrastive Loss), the other is **Inter Contrastive
​       Loss** (which deals with interactions of prototypes between intent detection and slot filling, and loss $\mathcal{L}^\mathcal{R}\_\mathcal{i}$ of the set of related slots to the i-th intent is calculated, and loss $\mathcal{L}^\mathcal{U}\_\mathcal{i}$ of that except with unrelated slots, the two terms is summed as the final Inter Contrastive 
​       loss), **Final Contrastive Loss** is the sum of Intra and Inter Contrastive Loss

​       **Highlights**

​                The Intra and Inter Contrastive Loss is quite fresh and suits the initiative of bridging the two metric spaces, however, neither 
​       the modified Contrastive Loss nor the Merged prototypes actually make it fit really well to the motivation of 'Bridging', and of course I 
​       was hoping to see more dissimilar tasks can be tackled using the idea of 'Bridging the Metric Space', this work definitely could have 
​       done much better by digging deeper.

- **Architecture**


<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic9.png?token=AOW4OJQKXG2LOS6JAEAO2CTBQ7EFC" /></div>




- **Paper**

###                                **CAN: Constrained Attention Networks for Multi-Aspect Sentiment Analysis**   [[paper]](https://aclanthology.org/D19-1467.pdf)

- **Background and motivation**

​       For multi-aspect sentiment analysis, the traditional attention on sentiment words for different aspects tends to be mixed up thus 
​       introducing noise for classification. In this work, the author divide sentences into two categories: **overlapping** and **non-**
​       **overlapping**. 
​       **Overlapping** means sentiment words for different aspects may overlap, e.g.: '*the food and service are good there.*' where the 
​       sentiment word for '*food*' and '*service*' is both '*good*', **non-overlapping** means sentiment words for different aspects aren't 
​       overlapped, e.g.: '*I like the food and the service is OK as well.*' where the sentiment word for '*food*' is '*like*' and for '*service*' is '*OK*'. 
​       And this work only tackles the mixed attention problem for the non-overlapping sentences, with the assumption of attentions for 
​       different sentiment words should be sparse, there are two interesting regularizers are proposed: 1. **Sparse Regularizer**;  2. 
​       **Orthogonal Regularizer** to help attend to different words for different aspects in non-overlapping sentences.  

- **Solution**

​       For Sparse Regularizer, the regularization term is: $\mathcal{R}\_\mathcal{s}$= $\vert \sum_{l=1}^L\alpha_{kl}^2-1\vert$, where $\sum_{l=1}^L\alpha_{kl}=1$ whose elements are all positive 
​       (attention score), by minimizing this regularizer, the elements will tend to a distribution like: {0, 0, ..., 1, 0, ..., 0} with the sentiment 
​       word having the largest attention score 1 and other words as 0. For Orthogonal Regularizer, $\mathcal{R}\_\mathcal{o}=\|\mathcal{M}^T\mathcal{M}-\mathcal{I}\|\_2$, where 
​       $\mathcal{M}\in\mathbb{R}^{\mathcal{k}\times\mathcal{L}}$, whose first dimension stands for K aspects in one sentence, second dimension stands for the length L for that 
​       sentence, in that sense, each row of the matrix is actually attention scores for one aspect for the sentence, by minimizing this 
​       regularizer, the attention score for different aspects tend to be close to 0 (since they are not on the diagonal of $\mathcal{M}^T\mathcal{M}$), thus 
​       producing orthogonality for different aspect attention scores.

​       **Highlights**

​               I personally like very much **the idea of making attention score sparse to attend to different sentiment words** by the 
​       proposed two regularizers, however if observe carefully, the Orthogonal Regularizer actually did the job for the Sparse Regularizer 
​       on its diagonal elements, so the Sparse Regularizer is some kind of redundant, which is also verified by the ablation experiments 
​       where the Orthogonal Regularizer performs better than the Sparse Regularizer. 

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic10.png?token=AOW4OJW5M3ED4T4TWZWYSULBQ7EFU" /></div>



- **Paper**

###                                **Contrast Learning Visual Attention for Multi Label Classification**   [[paper]](https://arxiv.org/pdf/2107.11626.pdf)

- **Background and motivation**

​       Previous image classification work simply apply contrastive learning to multi-label image classification which may only achieve OK 
​       results. By using label information to help the model to focus on different semantic components of the image, the proposed 
​       framework achieved better performance on multi-label image classification.

- **Solution**

​       The image is firstly encoded with CNN-based model (e.g.: ResNet) and then later the encoded features of the image are later 
​       together with a randomly initialized class-embedding matrix $\mathcal{U}\in\mathbb{R}^{\mathcal{L}\times\mathcal{C}}$ whose raw is the embedding for one class. Later the matrix
​       is used as Query, and the encoded features are treated as Key and Value for a Multi-Attention block, the outputs from the Multi-
​       Attention Blocks are then later used for classification. The framework adopts a two stage training, in stage one, the model is trained 
​       with only binary cross entropy loss to make the model to learn a task specified $\mathcal{U}$, in stage two, a contrastive loss is added to help 
​       fine-tune the class-embedding matrix $\mathcal{U}$ to achieve promising results.

​       **Highlights**

​              The randomly initialized class-embedding matrix is OK however **the performance of one task may be closely related to how the weights are initialized** (e.g.: 'RandomUniform' or 'RandomNormal') , thus this framework may not be very robust for different 
​       image classification datasets, like how I mentioned in the **Highlights** of [**Joint Embedding of Words and Labels for Text
​       Classification**](#3).
​       The usage of simply a cos similarity of something distance/similarity alike calculation for the initialization of the class-embedding 
​       matrix $\mathcal{U}$ will be definitely more convincing. (**Though this work used train + fine-tune to overcome the possible instability 
​       caused by the initialization of the class-embedding matrix $\mathcal{U}$, still, not a very optimal way in my opinion.**)

- **Architecture**


<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic11.png?token=AOW4OJUGLPNULTYUMZP3HD3BQ7EGG" style="zoom:75%;" /></div>




- **Paper**

###                                **Inconsistent Few-Shot Relation Classification via Cross-Attentional Prototype Networks with Contrastive Learning**    [[paper]](https://arxiv.org/abs/2110.08254)

- **Background and motivation**

​       For standard Few-Shot learning, when training/testing, it always sticks to the setting of N-way K-shot, however in real life scenario, 
​       it's hard to keep the N and K invariable, it's more likely that the N and K for two different meta-tasks ( during training/testing) will be 
​       different as well. Under the inconsistent scenario, the author proposed a prototype based network, together with cross-attention as 
​       well as contrastive learning to tackle the inconsistent N/K problem for Few-Shot Learning.

- **Solution**

​       To tackle the inconsistent N/K problem, a cross-attention is performed to learning better prototype representations (since N/K is 
​       inconsistent, for some meta-tasks it will be harder for the meta-learner to learn good prototypes with more classes and less 
​       examples). For attention *Support* $\rightarrow$ *Query*, a distribution $\mathcal{d}^\mathcal{i}\_\mathcal{r}$ is calculated for every support instance $\mathcal{s}^\mathcal{i}\_\mathcal{r}$ with all the query 
​       instances, and a distance loss $\mathcal{L}\_\mathcal{dist}$ is minimized when the closer the intra-class instances and the further the inter-class instances 
​       are, the smaller the loss is. For attention *Query* $\rightarrow$ *Support*, an attention mechanism similar to xxx(links here) is calculated to 
​       weight the support instance $\mathcal{s}^\mathcal{i}\_\mathcal{r}$. With the main Cross-Entropy loss and the distance loss, a contrastive loss is calculated over $\mathcal{s}^\mathcal{i}\_\mathcal{r}$ where distance between two instances from the same class will be small while that of different classes will be large.       

​       **Highlights**

​                **The setting of inconsistent N/K** is definitely the core idea of the paper, however, in my eyes, **how it solved this problem is 
​       controversial**, since it used the information from the query to do attention on support set to get the prototype, which can be viewed 
​       as the examples for the support set are implicitly increased (query information is involved, in that sense, I think it should be carefully 
​       judged the way how [Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification](#4) did support set 
​       attention as well), and when comparing, the model is directly compared to prototype-based networks **without** doing cross attention 
​       and only achieved a few performance improvements (thus hard to define whether this architecture is really good at solving the 
​       inconsistent problem or the prototype-based model with cross-attention can still perform well), also, **the experiments are 
​       incomplete**, where for inconsistent N/K settings, the K/N did not cover a reasonable range. (e.g.: when K for training are 5, 10, 20, 
​       K for testing are 1, 5, 10, 20, didn't experiment with the setting of K for testing greater than 20)

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic12.png?token=AOW4OJSKLNQWQ73I2QDI3A3BQ7EGU" style="zoom:75%;" /></div>




- **Paper**

###                                **Supervised Contrastive Learning***    [[paper]](https://arxiv.org/abs/2004.11362)    *<u>(favorite paper in this summary)</u>*

- **Background and motivation**

​       The supervised contrastive learning actually makes use of the class information to modify the previous self-supervised contrastive 
​       learning, with the intuition that given instance labels, those of the same label should be closer to each other.

​       **Highlights**

​              The usage of label information, **simple yet efficient** ! And the proof in the Supplementary is excellent as well.

- **Solution**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic13.png?token=AOW4OJQOJPOZ46RBQNRKFC3BQ7ER2" style="zoom:75%;" /></div>




- **Paper**

###                                **Multi-Label Few-Shot Learning for Aspect Category Detection**   [[paper]](https://arxiv.org/abs/2105.14174)

- **Background and motivation**

​        A new dataset in Multi-Aspect Category Detection for Few-Shot learning is released in this work, and **to alleviate noise meanwhile learn expressive and distinct representation to do classification**, an attention based model is proposed.

- **Solution**

​       Attention is done on support set texts as well as on query sentences, which is with the function of  reducing noise thus help to learn 
 ​      a better prototype representation/ a clearer query representation, Euclidean distance is computed between prototypes and query 
​       representations to output the final classification probability. 

​       **Highlights**

​               The idea of reducing noise on support and query set is quite cool, and the release of a new dataset is a good supplementary 
​       as well, however the way the author does attention on support and query set is inconsistent and a little bit weird.

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic14.png?token=AOW4OJUJ77XBSGUEI7P2B2LBQ7EIO" style="zoom:85%;" /></div>




- **Paper**

###                                **Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification**   [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/4604)<a name='4'></a>

- **Background and motivation**

​        Previous work of Relation Classification relies on distant supervision (e.g.: the usage of database), this work treat the task as Few-
 ​       Shot Learning problem and uses attention to produce **noise-proof** hybrid prototypes to do classification.

- **Solution**

​        Two different attention mechanisms (Instance-level attention and Feature-level attention) are used to reduce noise, for Instance-
 ​       level attention, query texts are used to help calculate attention weights over the support sentences to get the prototypes, and for 
​        Feature-level attention, a scoring vector $\mathcal{z}\_i$ for each class is computed using K sentences of that class. Later the distance between 
​        the encoded query texts and the prototypes is computed with the help of the scoring vector $\mathcal{z}\_i$.

​        **Highlights**

 ​               The idea of applying Few-Shot learning to do relation classification is stunning, however the way support attention 
 ​       $\alpha=softmax(\mathcal{e})$ (where $\mathcal{e}\_j=sum\{tanh(g(x^j_i)\odot g(x))\}$, $x^j_i$ is the encoded support sentence and $x$ stands for all the encoded 
​        query texts) is calculated has introduced extra query information to help the support set to learn a better prototype, and I personally 
​        do not recommend doing support set's attention with query information.

- **Architecture**

<div align=center><img src="https://raw.githubusercontent.com/A-Chicharito-S/img/paper_summary_1/pic15.png?token=AOW4OJQ5GJ27U47UB37WEVLBQ7EJE" style="zoom:75%;" /></div>

