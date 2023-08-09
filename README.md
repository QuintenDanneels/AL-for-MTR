# AL-for-MTR
<p>Recent technical innovations have led to an exponential increase in data production. This creates problems in domains where the data has to be labeled by domain experts and where this entails a substantial cost so that it can be used in supervised machine learning algorithms. **Active learning (AL)** is a subfield in machine learning that solves this problem. It allows the models to be built on fewer but more representative data instances. Although a fair amount of research has already been conducted in the field of AL, there is little research performed on AL in the domain of **multi-target regression (MTR)**, where one wants to predict multiple targets based on multiple features. In this thesis, we offer a public framework with baseline and state-of-the-art algorithms. We also present a new algorithm called **Query-By-Committee with Random Forest (QBC-RF)**, which has been validated on various MTR benchmark datasets. Our results show that QBC-RF has better predictive performance results than its competitors.<\p>

<p>This repository contains the following folders:
1. **Literature**: literature about active learning and multi-target regression.
2. **Datasets**: different datasets used for MTR as well as seperate folders containing training, unlabbeled pool and test sets.
3. **Scripts**: python scripts for the subsets generation, active learning algorithms and Friedman-Nemenyi test.
4. **Results**: results for all the different AL methods. 
5. **Presentations**: presentations used for my thesis. </p>
