---
title: 'Neural Nets Are Not All You Need: Evaluating the Effects of Deep Learning on Transcriptomic Analysis'
keywords:
- machine learning
- science of science
- reproducibility
- citation network analysis
lang: en-US
date-meta: '2022-12-01'
author-meta:
- Benjamin J. Heil
header-includes: |-
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta name="dc.title" content="Neural Nets Are Not All You Need: Evaluating the Effects of Deep Learning on Transcriptomic Analysis" />
  <meta name="citation_title" content="Neural Nets Are Not All You Need: Evaluating the Effects of Deep Learning on Transcriptomic Analysis" />
  <meta property="og:title" content="Neural Nets Are Not All You Need: Evaluating the Effects of Deep Learning on Transcriptomic Analysis" />
  <meta property="twitter:title" content="Neural Nets Are Not All You Need: Evaluating the Effects of Deep Learning on Transcriptomic Analysis" />
  <meta name="dc.date" content="2022-12-01" />
  <meta name="citation_publication_date" content="2022-12-01" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Benjamin J. Heil" />
  <meta name="citation_author_institution" content="Genomics and Computational Biology Graduate Group, University of Pennsylvania" />
  <meta name="citation_author_orcid" content="0000-0002-2811-1031" />
  <meta name="twitter:creator" content="@autobencoder" />
  <link rel="canonical" href="https://greenelab.github.io/ben_heil_dissertation/" />
  <meta property="og:url" content="https://greenelab.github.io/ben_heil_dissertation/" />
  <meta property="twitter:url" content="https://greenelab.github.io/ben_heil_dissertation/" />
  <meta name="citation_fulltext_html_url" content="https://greenelab.github.io/ben_heil_dissertation/" />
  <meta name="citation_pdf_url" content="https://greenelab.github.io/ben_heil_dissertation/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://greenelab.github.io/ben_heil_dissertation/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://greenelab.github.io/ben_heil_dissertation/v/8cdf9e8f9f1efd3e7efbd1789620f5082cc17b2c/" />
  <meta name="manubot_html_url_versioned" content="https://greenelab.github.io/ben_heil_dissertation/v/8cdf9e8f9f1efd3e7efbd1789620f5082cc17b2c/" />
  <meta name="manubot_pdf_url_versioned" content="https://greenelab.github.io/ben_heil_dissertation/v/8cdf9e8f9f1efd3e7efbd1789620f5082cc17b2c/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography:
- content/manual-references-hgp.json
- content/manual-references-indices.json
- content/manual-references-linear.json
- content/manual-references-mousiplier.json
- content/manual-references-repro.json
- content/manual-references.json
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
manubot-clear-requests-cache: false
...

## Acknowledgements

I would not have reached this point without the support of many people. 
First I would like to thank my mentor Casey Greene for helping me grow from a first-year grad student with aspirations of diagnosing all human disease with a cleverly designed model to a wisened (or maybe wizened) PhD candidate who believes that data is paramount. 
I still remember reading papers as an undergrad trying to better understand what was going on at the intersection of computational biology and machine learning and wondering “What is the University of Pennsylvania and why does this Casey guy’s papers keep showing up in my searches?” 
Thank you to my thesis committee: Marylyn Ritchie, Russ Altman, Konrad Kording, and Kai Wang. 
Your feedback has helped keep my research from going off the rails. 
Thank you as well to Greenelab members past and present. 
From grilling me to help prepare for prelims, to going on adventures with me in Colorado, to giving me tips on where to find free food, you’ve all helped me to better understand science and what it means to be a scientist. 
Thanks to Shuo Zhang and Liz Heller for collaborating with me on MousiPLIER, the project would not have been possible without you. 
I would also like to thank John Holmes for agreeing to be my advisor at Penn when Greenelab moved west to Colorado. 
In addition, I’d like to thank the GCB administration, especially Maureen Kirsch, Anne-Cara Apple, and Ben Voigt. 
You all do a good job of looking after students and making sure we don’t fall through the cracks due to conditions beyond our control.

I’ve also been helped through grad school by many people outside of academia. 
Thanks Mom, Dad, Nana, Mary, Wes, and Sujin, your support has meant a lot even if you don’t always understand what I’m talking about. 
Thanks as well to Rachel Ungar, and sorry that we didn’t get an opportunity to collaborate (yet?) 
If you hadn’t teamed up with me in the audacious plan to get internships at the NIH after sophomore year, I wouldn’t be where I am today. 
Thanks to my friends in Philly and in Texas for convincing me to get outside the lab and have fun on occasion, and for giving conflicting advice on whether or not I should drop out of grad school. 
Finally, thank you Sydney for helping me see that there is a world going on outside of the small bubble I interact with on a daily basis. 
I know that living with a PhD candidate has been frustrating at times, especially as I’ve gotten closer to defending and therefore progressively less interesting. 
I’m tempted to come up with something witty to write here, but you’d probably prefer sincerity, so: thank you.


## Abstract {.page_break_before}

Technologies for quantifying biology have undergone significant advances in the past few decades, leading to datasets rapidly increasing in size and complexity.
At the same time, deep learning models have gone from a curiosity to a massive field of research, with their advancements spilling over into other fields.
Machine learning is not new to computational biology, as machine learning models have been used frequently in the field to account for the aforementioned size and complexity of the data.
This dissertation asks whether the paradigm shift in machine learning that has led to the rise of deep learning models is causing a paradigm shift in computational biology.
To answer this question, we begin with chapter 1, which gives background information helpful for understanding the main thesis chapters.
We then move to chapter 2, which discusses standards necessary to ensure that research done with deep learning is reproducible.
We continue to Chapter 3, where we find that deep learning models may not be helpful in analyzing expression data.
In chapters 4 and 5 we demonstrate that classical machine learning methods already allow scientists to uncover knowledge from large datasets.
Then in chapter 6 we conclude by discussing the implications of the previous chapters and their potential future directions.
Ultimately we find that while deep learning models are useful to various subfields of computational biology, they have yet to lead to a paradigm shift.


# Chapter 1: Background

This chapter was prepared for this dissertation to provide background information and context for the dissertation as a whole.
Parts of the "Applications of Machine Learning in Transcriptomics" were written for the class GCB752.

**Contributions:**  
I was the sole author for this chapter.
Some of the "Applications of Machine Learning in Transcriptomics" section was edited based on feedback from Kara Maxwell.

## Introduction
                                                                                                                                                                                                            
As computational biologists, we live in exciting times.
Beginning with the Human Genome Project [@doi:10.1038/35057062], advancements in technologies for biological quantification have generated data with a scale and granularity previously unimaginable [@doi:10.1016/j.cell.2015.05.002; @doi:10.7554/eLife.21856; @doi:10.1101/gad.281964.116].

Concurrently with the skyrocketing amounts of data, the advent of deep learning has generated methods designed to make sense of large, complex datasets.
These methods have led to a paradigm shift [@isbn:9780226458113] in machine learning, creating new possibilities in many fields and surfacing new phenomena unexplained by classical machine learning theory [@doi:10.1038/nature16961; @arxiv:2112.10752; @arxiv:1912.02292; @arxiv:2201.02177].

The field of computational biology has long used machine learning methods, as they help cope with the scale of the data being generated.
Accordingly, problem domains in computational biology that map well to existing research in deep learning have adopted or developed deep learning models and seen great advances [@doi:10.1007/978-3-319-24574-4_28; @doi:10.1038/s41586-021-03819-2].
Previous applications of classical machine learning to the field of transcriptomics have been successful.
Two of the scientists who wrote the book [@isbn:0387848576] on machine learning have even written papers [@doi:10.1073/pnas.082099299; @doi:10.1093/biostatistics/kxl002; @doi:10.1186/gb-2000-1-2-research0003; @doi:10.1093/biostatistics/kxl005] analyzing gene expression.
However, the data itself is not well-suited to deep learning methods.

This dissertation explores whether the paradigm shift in machine learning will spill over to transcriptomics.
That is to say, have deep learning techniques fundamentally changed transcriptomics, or are they incremental improvements over existing methods?
Our thesis is that while deep learning provides valuable tools for analyzing biological datasets, it does not necessarily change the field on a fundamental level.

We begin with chapter 1, which gives background information on previous research for the other thesis chapters.
We then move to chapter 2, which discusses standards necessary to ensure that research done with deep learning is reproducible.
We continue to Chapter 3, where we find that deep learning models may not be helpful in analyzing expression data.
In chapters 4 and 5 we demonstrate that classical machine learning methods already allow scientists to uncover knowledge from large datasets.
Finally, in chapter 6 we conclude by discussing the implications of the previous chapters and their potential future directions.



## Applications of machine learning in transcriptomics

The human transcriptome provides a rich source of information about both healthy and disease states.
Not only is gene expression information useful for learning novel biological phenomena, it can also be used to diagnose and predict diseases.
These predictions have become more powerful in recent years as the field of machine learning has developed more methods.
In this section we review machine learning methods applied to predict various phenotypes from gene expression, 
with a focus on the challenges in the field and what is being done to overcome them.
We close the review with potential areas for future research, as well as our perspectives on the strengths and weaknesses of supervised learning for phenotype prediction in particular.

### Introduction
Over the past few decades a number of tools for measuring gene expression have been developed.
As proteomics is currently difficult to do at a large scale, gene expression quantification methods are our best way to measure cells’ internal states.
While this wealth of information is promising, gene expression data is more difficult to work with than one might think.
The high dimensionality and instrument-driven variation require sophisticated techniques to separate the signal from the noise.

One such class of techniques is the set of methods from machine learning.
Machine learning methods depend on the assumption that there are patterns in data that can be learned to make predictions about future data.
Luckily, different people respond to the same disease in similar ways (for some diseases).
Learning genes that indicate an inflammatory response, for example, can help a machine learning model learn the difference between healthy and diseased expression samples.

There are many varieties of machine learning algorithms, so the scope of this paper is limited to analysis of supervised machine learning methods for phenotype prediction.
Supervised machine learning is a paradigm where the model attempts to predict labels.
For example, a model that predicts whether someone has lupus based on their gene expression data [@doi:10.1038/s41598-019-45989-0] is a supervised learning model.
In contrast, techniques for grouping data together without having phenotype labels are called unsupervised methods.
While these methods are also commonly used in computational biology [@doi:10.1093/bioinformatics/btz338; @doi:10.48550/arXiv.1802.03426; @doi:10.1093/bioinformatics/btz769], we will not be discussing them here.

The purpose of this review is to explain and analyze the various approaches that are used to predict phenotypes.
Each section of the review is centered around one of the challenges ubiquitous in using supervised machine learning techniques on gene expression data.
We hope to explain what has been tried and what the consensus for handling the challenge, if one exists.
The review will conclude with a section outlining promising new methods and areas where further study is needed.

If the field succeeds in addressing all the challenges, the payoffs will be substantial.
Being able to predict and diagnose diseases from whole blood gene expression is particularly interesting.
With sufficiently advanced analysis, invasive cancer biopsies might be able to be replaced with simple blood draws [@doi:@doi:10.1093/bioinformatics/btz769].
If not, there are already diagnostics that predict various cancer aspects from biopsy gene expression [@doi:10.1126/scitranslmed.aay1984].
It may also be possible to diagnose common diseases based on blood gene expression [@doi:10.1038/modpathol.2008.54; @doi:10.1212/WNL.0000000000004516; @doi:10.1128/JCM.01990-15; @doi:10.1016/j.ebiom.2016.03.006], or even rare ones [@doi:10.1002/art.22981].

The techniques for measuring gene expression and for analyzing it have changed dramatically over the past few decades.
This sections aims to explain what some of those changes are and how they affect phenotype prediction.

### Gene expression
Gene expression measurement methods have three main categories.
This first to be created is the gene expression microarray.
In a microarray, RNA is reverse transcribed to cDNA, labeled with fluorescent markers, then hybridized to probes corresponding to parts of genes.
The amount of fluorescence is then quantified to give the relative amount of gene expression for each gene.
While early microarrays had fewer genes and gene probes [@doi:10.1126/science.1990438], more modern ones measure tens of thousands of genes [@doi:10.1038/nbt1296-1675].

While microarrays are useful, decreases in the price of genetic sequencing have made bulk RNA sequencing (RNA-seq) more common.
In RNA-seq, cDNA molecules are sequenced directly after being reverse transcribed from mRNA.
These cDNA fragments are then aligned against a reference exome to determine which gene, if any, each fragment maps to.
The output of the bulk RNA-seq pipeline is a list of genes and their corresponding read counts.
While there is not gene probe bias like in microarrays, RNA-seq has its own patterns of bias based on gene lengths and expression levels [@doi:10.1371/journal.pone.0078644].
Bulk RNA-seq is also unable to resolve heterogeneous populations of cells, as it measures the average gene expression of all of cells in the sample.

Fairly recently a new method was developed called single-cell RNA sequencing.
True to its name, single-cell sequencing allows gene expression to be measured at the individual cell level.
This increase in precision is accompanied by an increase in data sparsity though, as genes expressed infrequently or at low levels may not be detected.
The sparsity of single-cell data has led to a number of interesting methods, but as we worked with bulk RNA-sequencing single-cell papers will largely be absent from this review.

### Machine Learning
Machine learning has undergone a paradigm shift in the past decade, beginning with the publication of the AlexNet paper in 2012 [@doi:10.1145/3065386].
For decades random forests and support vector machines were the most widely used models in machine learning.
This changed dramatically when the AlexNet paper showed that neural networks could vastly outperform traditional methods in some domains [@doi:10.1145/3065386].
The deep learning revolution quickly followed, with deep neural networks becoming the state of the art in any problem with enough data [@doi:10.48550/arXiv.1808.09381; @arxiv:1910.10683v3; @arxiv:1505.04597; @doi:10.1038/s41586-021-03819-2].

The implications of the deep learning revolution on this paper are twofold.
First, almost all papers before 2014 use traditional machine learning methods, while many papers after use deep learning methods.
Second, deep neural networks’ capacity to overfit the data and fail to generalize to outside data are vast.
We’ll show throughout the review various mistakes authors make because they don’t fully understand the failure states of neural networks and how to avoid them.

### Dimensionality Reduction
The most obvious challenge in working with gene expression data is its high dimensionality.
That is to say that the number of features (genes) in a dataset is typically greater than the number of samples.
It is common for an analysis to have tens of thousands of genes, but only hundreds (or tens) of samples.
Because even simple models struggle under such circumstances, it is necessary to find a representation of the data that uses fewer dimensions.

In the traditional machine learning paradigm, this is done via manual or heuristic feature selection methods.
Such methods tend to use a criterion like mutual information to select a subset of genes for the analysis [@doi:10.1007/s00521-013-1368-0].
In one of the earliest papers in this review, Li et al. try a eight different methods from statistics and machine learning to see if any one in particular outperformed the others [@doi:10.1093/bioinformatics/bth267].
Ultimately they found that no individual method rose to the top, and that the performance of different methods varies depending on the problem.

A number of other papers since then have also used manual methods.
Grewal et al. chose a subset of genes from COSMIC [@doi:10.1093/nar/gkp995] for training, but found that their model performed better when using all genes instead of just a subset [@doi:10.1001/jamanetworkopen.2019.2597].
Chen et al. used a different gene set.
They selected the LINCS 1000 gene set [@doi:10.1016/j.cell.2017.10.049] for an imputation method, as the LINCS landmark genes are highly correlated with the genes they were trying to impute [@doi:10.1093/bioinformatics/btw074].

Gene subsets can be based on prior knowledge of gene regulatory networks as well [@doi:10.1093/bioinformatics/btu272; @doi:10.1038/s41598-018-19635-0].
While very interpretable, these methods do not necessarily lead to increased performance in phenotype predictions [@doi:10.1186/1471-2105-13-69].
However, such methods can be useful in their own right.
PLIER (and the associated MultiPLIER framework) use prior knowledge genes to guide the latent variables learned by a matrix factorization technique [@doi:10.1038/s41592-019-0456-1; @doi:10.1016/j.cels.2019.04.003].
The resulting latent variables can then be used in differential expression analyses in lieu of raw gene counts, allowing dimensionality reduction while guiding the learned variables towards biological relevance.

Selecting gene subsets via a heuristic or a machine learning model is also popular.
Sevakula et al. use decision stumps to select features then use a stacked autoencoder-type architecture to further compress the representation [@doi:10.1109/TCBB.2018.2822803].
Xiao et al. did something similar where they reduced the data to only genes were differentially expressed between their conditions of interest, then used a stacked autoencoder architecture [@doi:10.1016/j.cmpb.2018.10.004].
Instead of looking at raw differential expression, Dhruba et al. used another subsetting method called ReliefF [@doi:10/fdm4r3] to find the top 200 genes for their source and target dataset, then kept the intersection for use in their model [@doi:10.1186/s12859-018-2465-y].
More recently, Li et al. used a genetic algorithm for feature selection [@doi:10.1186/s12864-017-3906-0].

Not all papers use a subset of the original genes in their analysis, however.
It is fairly common in recent years for authors to transform the data into a new lower dimensional space based on various metrics.
This used to be done via principle component analysis (PCA), a method that performs a linear transformation to
maximize the variance explained by a reduced number of dimensions [@doi:10.1016/j.compbiomed.2014.09.008; @fakoor2013].
Now scientists typically use different types of autoencoders, which learn a nonlinear mapping from the original space to a space with fewer dimensions.
Deepathology uses variational [@arxiv:1312.6114] and contractive [@doi:10.1007/978-3-642-23783-6_41] autoencoders in their model [@doi:10.1038/s41598-019-52937-5], while Danaee et al. used a stacked denoising autoencoder [@vincent2010; @doi:10.1142/9789813207813_0022].
Both papers compared their autoencoder dimensionality reduction to that of PCA and found that it performed better.
Danaee found that kernel PCA, a nonlinear version of PCA performed equivalently though.

It is also possible to use regularization methods to perform dimensionality reduction.
While they do not influence the nominal dimensionality of the data, they reduce the effective dimensionality by putting constraints on the input data or the model.
For example, SAUCIE uses an autoencoder structure, but combines it with a number of exotic regularization methods to further decrease the effective dimensionality of their data [@doi:10.1038/s41592-019-0576-7].
In DeepType, Chen et al. use a more conventional elastic net regularization [@zou2005] to induce sparsity in the first level of their network under the assumption that most genes’ expression will not affect a cancer’s subtype [@doi:10.1093/bioinformatics/btz769].

Ultimately, there is no clear consensus in which dimensionality reduction methods perform the best.
Among the methods that transform the data there is a small amount of evidence that nonlinear transformations outperform linear ones, but only a few studies have tried both.
Going forward, a systematic evaluation of gene selection and dimensionality reduction methods on a variety of problems could be a huge asset to the
field.

### Evaluating Model Performance
Validation is another important consideration in phenotype prediction.
The gold standard of validation would be a knockout and rescue assay demonstrating that the predicted mechanism or expression relationship truly exists.
Since machine learning models make predictions of nonlinear relationships between thousands of genes, however, such validation isn’t feasible.
Instead scientists evaluate their models’ efficacy by testing their performance on data they didn’t train them on.
Test datasets can be built in different ways, assorting roughly into three tiers based on their external validity.

The most basic method is referred to as cross-validation.
In cross-validation, the training data is split into a training and validation dataset.
The model is trained on the training dataset, then its performance is measured on the validation dataset.
Typically this is done with a process called five-fold cross-validation, where the process is repeated five times on five different ways of splitting up the training data.
This method is common [@doi:10.1186/s12859-018-2465-y; @doi:10.1186/s12864-017-3906-0; @doi:10.1142/9789813207813_0022], but isn’t really a rigorous evaluation.
Because the same dataset is used for both selecting a model and measuring performance, the data can ’go stale’ when you test several models [@doi:10.1016/j.neubiorev.2020.09.036].
In the extreme case, it is possible to get 100% accuracy by testing random prediction schemes on the data.

In order to keep data fresh, some researchers use a more rigorous method called a held out test set[@doi:10.1038/s41598-019-52937-5; @doi:10.1109/tcbb.2018.2822803].
In the held out test set paradigm, a portion of the dataset is set aside and effectively put in a locked box until the end of the analysis.
Once the model architecture, hyperparameters, and dimensionality reduction decisions are all made via cross-validation on the training data, the lock box can be opened and the data within used for evaluation.
As the lock box data is only used once, it has no risk of becoming stale due to multiple testing.
The only drawback to this method is that is depends on the assumption that the data in the real world is distributed the same as the data in your training set.

The best (and most difficult) way to evaluate a model is by using an independent dataset.
Ideally, an independent dataset is created by a different group or on a different expression quantification platform.
For example, once their model was trained, Chen et al. evaluated their model on a dataset from GEO, a dataset from GTEx, and a cancer cell line [@doi:10.1093/bioinformatics/btw074].
It is also possible to use combinations of validation methods.
In their paper Grewal et al. used a held-out section of their original data, then went on to evaluate their model in an independent dataset [@doi:10.1001/jamanetworkopen.2019.2597].
Similarly, Malta et al. used cross-validation initially, but then evaluated their model on an external microarray dataset to ensure their data wasn’t stale [@doi:10.1016/j.cell.2018.03.034].
Likewise, Deng at al. initially benchmark their model on various simulated data sets, but then go on to validate their model on real data [@doi:10.1101/315556].

Ultimately researchers work with what they have, and it’s not always possible to acquire an independent dataset.
That being said, it is always worth keeping the different tiers of external validity in mind when evaluating papers that use machine learning.

### Transfer Learning
Transfer learning is a field of machine learning that uses information from outside of the training dataset to improve model performance.
Techniques from the field of transfer learning are particularly useful in the domain of gene expression, because there are large databases like GEO and TCGA that contain data that may be useful in prediction tasks.
In this section we’ll focus in on two types of transfer learning that are particularly useful: multitask learning and semi-supervised learning.

Multitask learning involves training a model on multiple problems in order to improve the model’s performance on a problem of interest.
As gene expression patterns can be shared across diseases [@doi:10.18632/oncotarget.26773; @doi:10.1038/s41598-017-02325-8], the extra data can help increase the model’s power.
For example, instead of training a model to learn one drug response at a time, Yuan et al.
had better results predicting all the drugs in their dataset simultaneously [@doi:10.1038/srep31619].
Similarly, Deepathology predicts tissue type, disease, and miRNA expression simultaneously [@doi:10.1038/s41598-019-52937-5].
It is worth noting that multitask learning works best when using a deep learning model.
When using standard machine learning it is necessary to perform some difficult data transformation to do classification on multiple classes [@doi:10.1093/bioinformatics/bth267].

Where supervised learning uses entirely labeled data, semi-supervised learning takes advantage of unlabeled data as well.
The most popular way of doing semi-supervised learning is to use an autoencoder structure to initialize your model’s weights.
Where most models begin training with a randomly initialized set of weights, it is possible to initially train a neural network to create a compressed representation of the input data (an encoding).
The weights that it learns in the process often turn out to be a better initialization when the labeled training data is finally brought in.
There are a number of ways to perform the autoencoding step.
Instead of training all the layers of the network simultaneously, it is possible to train one layer to create the encoding at a time [@doi:10.1109/tcbb.2018.2822803; @doi:10.1016/j.cmpb.2018.10.004].
This is referred to as a stacked autoencoder.
One can also train the whole network at the same time, as Danaee et al do with their denoising autoencoder [@doi:10.1142/9789813207813_0022].
Not all methods are autoencoder-based though.
Dhruba et al. develop their own semi-supervised learning process that teaches a model to learn a latent space between classes [@doi:10.1186/s12859-018-2465-y].

### Deep Learning vs Classical ML
Recent years have seen a dramatic shift towards deep learning methods.
It is not immediately clear, however, whether this is a good decision for problems without giant datasets.
While some argue that deep learning is overrated and simpler models should be used instead [@doi:10.1007/978-3-030-01768-2_25; @arxiv:1911.00353], others find that deep learning outperforms even domain specific models [@doi:10.1186/s13321-017-0232-0; @doi:10.1016/j.jbi.2018.04.007].

Because it is unclear which type of model will perform best on which dataset, it is important to try both simple and complex models.
In the Deepathology paper, Azarkahlili et al. found that their deep neural networks outperformed decision tree, KNN, random forest, logistic regression, and SVM models [@doi:10.1038/s41598-019-52937-5 ].
Likewise, in gene expression imputation, Chen et al. found that their neural network classifier outperformed linear regression in 99.97 percent of genes and k-nearest neighbors in all genes [@doi:10.1093/bioinformatics/btw074].
On the other hand, Grewal et al. tried multiple methods and found they work roughly the same [@doi:10.1001/jamanetworkopen.2019.2597].
They settled this by combining a few different models into an ensemble.

Due to technical considerations [@doi:10.1016/j.cell.2018.03.034] or other reasons, some authors only evaluate a single model [@doi:10.1016/j.cmpb.2018.10.004].
While this simplifies the analysis for their papers, it makes it unclear whether they could have done better with a different model.
This is particularly important for authors who are using deep learning models, because simpler models tend to be much more interpretable.

In chapters 3 and 4, we apply machine learning models to transcriptomic data.
Chapter 3 has us comparing linear and deep learning models and showing that the linear models perform at least as well as the neural networks.
Chapter 4 continues the idea by demonstrating that classical machine learning can be used to great effect on gene expression data.



## Citation indices

Over the past century quantifying the progress of science has become popular.
Even before computers made it easy to collate information about publications, work had already begun to evaluate papers based on their number of citations [@doi:10.1126/science.122.3159.108].
There is even a book about it [@isbn:1108492665].

Determining the relative "impact" of different authors and journals is a perennial question when measuring science.
One of the most commonly used metrics in this space is the h-index, which balances an author's number of publications with the number of citations each receives [@doi:10.1073/pnas.0507655102]. 
However, the h-index is not a perfect metric [@doi:10.1016/j.acalib.2017.08.013] and has arguably become less useful in recent years [@doi:10.1371/journal.pone.0253397].
Other metrics, like the g-index[@doi:10.1007/s11192-006-0144-7] and the i-10 index (https://scholar.google.com/), try to improve on the h-index by placing a higher weight on more highly cited papers.

There are metrics for comparing journals as well.
The Journal Impact Factor [@jif] is the progenitor journal metric, evaluating journals based on how many citations the average paper in that journal has received over the past few years.
Other measures use a more network-based approach to quantifying journals' importance.
The most common are Eigenfactor [@doi:10.5860/crln.68.5.7804] and the SCImago Journal Rank (https://www.scimagojr.com/), which use variations on the PageRank algorithm to evaluate the importance of various journals. 

Academic articles are arguably the main building blocks of scientific communication, so it makes sense to try to understand which ones are the most important.
Citation count seems like an obvious choice, but differences in citation practices between fields [@doi:10.1016/j.joi.2013.08.002] make it too crude a measure of impact.
Instead, many other metrics have been developed to choose which papers to read.

Many of these methods work by analyzing the graph formed by treating articles as nodes and citations as edges.
PageRank[@pagerank], one of the most influential methods for ranking nodes' importance in a graph, can also be applied to ranking papers [@doi:10.1073/pnas.0706851105].
It is not the only graph-based method, though.
Other centrality calculation methods, such as betweenness centrality, would make sense to use but are prohibitively computationally expensive to run.
Instead, methods like the disruption index [@doi:10.1038/s41586-019-0941-9] and its variants [@doi:10.1162/qss_a_00068] are more often used.

Some lines of research try to quantify other desirable characteristics of papers.
For example, Foster et al. claim to measure innovation by looking at papers that create new connections between known chemical entities [@doi:10.1177/0003122415601618].
Likewise, Wang et al. define novel papers as those that cite papers from unusual combinations of journals [@doi:10.1016/j.respol.2017.06.006].
The Altmetric Attention Score (https://www.altmetric.com/) goes even further, measuring the attention on a paper from outside the standard academic channels.

These metrics do not stand alone, however.
Much work has gone into improving the various methods by shoring up their weaknesses or normalizing them to make them more comparable across fields.
The relative citation ratio makes citation counts comparable across fields by normalizing it according to other papers in its neighborhood of the citation network [@doi:10.1371/journal.pbio.1002541].
Similarly, the source-normalized impact per paper normalizes article citation counts based on the total number of citations in the whole field [@doi:10.1016/j.joi.2010.01.002].
Several methods modify PageRank, such as Topical PageRank, which incorporates topic and journal prestige information into the PageRank calculation [@doi:10.1007/s11192-017-2626-1], and 
Vaccario et al.'s page and field rescaled PageRank, which accounts for differences between papers' ages and fields [@arxiv:1703.08071].
There are also several variants of the disruption index [@doi:10.1162/qss_a_00068].

Of course, these methods only work with data to train and evaluate them on.
We have come a long way from Garfield's "not unreasonable" proposal to aggregate one million citations manually [@doi:10.1126/science.122.3159.108].
These days we have several datasets with hundreds of millions to billions of references (https://www.webofknowledge.com, https://www.scopus.com  @doi:10.1007/s11192-019-03217-6).

Quantifying science could be better, however.
In addition to the shortcomings of individual methods [@doi:10.1523/JNEUROSCI.0002-08.2008; @doi:10.1016/j.wneu.2012.01.052; @doi:10.2106/00004623-200312000-00028], there are issues inherent to reducing the process of science to numbers.
To quote Alfred Korzybski, "the map is not the territory." 
Metrics of science truly measure quantitative relationships like mean citation counts, despite purporting to reflect "impact," "disruption," or "novelty."
If we forget that, we can mistake useful tools for arbiters of ground truth.

In chapter 5, we dive into one such shortcoming by demonstrating differences in article PageRanks between fields.
There we argue that normalizing out field-specific differences obscures useful signal and propose new directions of research for future citation metrics.


# Chapter 1: Reproducibility standards for machine learning in the life sciences

This chapter was originally published in Nature Methods as "Reproducibility standards for machine learning in the life sciences" by Benjamin J. Heil, Michael M. Hoffman, Florian Markowetz, Su-In Lee, Casey S. Greene, and Stephanie C. Hicks (https://doi.org/10.1038/s41592-021-01256-7).

**Contributions:**  
C.S.G. was responsible for conceptualization. B.J.H. was responsible for project administration. B.J.H. and S.C.H. wrote the original draft of the manuscript; and B.J.H., S.C.H., M.M.H., S.L., F.M. and C.S.G. contributed to reviewing and editing.


## Abstract
Establishing reproducibility expectations focused on data, models, and code will ensure that the life sciences community can trust machine learning analyses.

## Introduction
The field of machine learning has grown tremendously within the past ten years. 
In the life sciences, machine learning models are being rapidly adopted because they are well suited to cope with the scale and complexity of biological data. 
There are drawbacks to using such models though. 
For example, machine learning models can be harder to interpret than simpler models, and this opacity can obscure learned biases. 
If we are going to use such models in the life sciences, we will need to trust them. 
Ultimately all science requires trust [@isbn:9780691179001] — no scientist can reproduce the results from every paper they read. 
The question, then, is how to ensure that machine learning analyses in the life sciences can be trusted.

One attempt at creating trustworthy analyses with machine learning models revolves around reporting analysis details such as hyperparameter values, model architectures, and data splitting procedures. 
Unfortunately, such reporting requirements are insufficient to make analyses trustworthy. 
Documenting implementation details without making data, models, and code publicly available and usable by other scientists does little to help future scientists attempting the same analyses and less to uncover biases. 
Authors can only report on biases they already know about, and without the data, models, and code, other scientists will be unable to discover issues post-hoc.

For machine learning models in the life sciences to become trusted, scientists must prioritize computational reproducibility [@stodden2013]. 
Specifically, using published data, models, and code, other scientists must be able to obtain the same results as the original authors. 
With access to published data, models, and code, a researcher can confirm that a model functions and probe how the model functions. 
This means that using the published model a third party can examine for themselves the accuracy of reported results and biases in the model. 
Analyses and models that are reproducible by third parties can be examined in depth and, ultimately, become worthy of trust. 
To that end, the life science community should adopt norms and standards that underlie reproducible machine learning research.

## The menu
While many regard the computational reproducibility of a work as a binary property, we prefer to think of it on a sliding scale [@stodden2013] reflecting the time needed to reproduce. 
Published works fall somewhere on this scale, which is bookended by “forever”, for a completely irreproducible work, and “zero”, for a work where one can automatically repeat the entire analysis with a single keystroke. 
Since it makes little sense to impose a single standard dividing work into “reproducible” and “irreproducible”, we instead propose a menu of three standards with varying degrees of rigor for computational reproducibility:

The bronze standard: the authors make the data, models, and code used in the analysis publicly available. The bronze standard is the minimal standard for reproducibility. Without data, models, and code, it is not possible to reproduce a work.
The silver standard: in addition to meeting the bronze standard, (1) the dependencies of the analysis can be downloaded and installed in a single command, (2) key details for reproducing the work are documented, including the order in which to run the analysis scripts, the operating system used, and system resource requirements, and (3) all random components in the analysis are set to be deterministic. The silver standard is a midway point between minimal availability and full automation. Works that meet this standard will take much less time to reproduce than ones only meeting the bronze standard.
The gold standard: the work meets the silver standard, and the authors make the analysis reproducible with a single command. The gold standard for reproducibility is full automation. When a work meets this standard, it will take little to no effort for a scientist to reproduce it.

While reporting has become a recent area of focus [@doi:10.1038/s41591-020-1041-y; @doi:10.1093/jamia/ocaa088; @doi:10.1148/ryai.2020200029], excellent reporting is akin to a nutrition information panel. It describes information about a work, but is insufficient for reproducing the work. 
In the best case it provides a summary of what the researchers who conducted the analysis know about biases in the data, model limitations, and other elements. 
It does not, however, provide enough information for someone to fully understand how the model came to be. 
For these reasons, concrete standards for ensuring reproducibility should be preferred over reporting requirements.

## Bronze
### Data
Data are a fundamental component of analyses. 
Without data, models can not be trained and analyses can not be reproduced. 
Moreover, biases and artifacts in the data that were missed by the authors cannot be discovered if the data are never made available. 
For the data in an analysis to be trusted, they must be published. 

To that end, all datasets used in a publication should be made publicly available when their corresponding manuscript is first posted as a preprint or published by a peer-reviewed journal. 
Specifically, the raw form of all data used for the publication must be published. 
The way the bronze standard should be met depends on the data used. 
Authors should deposit new data in a specialist repository designed for that kind of data [@doi:10.1002/1873-3468.14067], when possible. 
For example, one may deposit gene expression data in the Gene Expression Omnibus [@doi:10.1093/nar/30.1.207] or microscopy images in the BioImage Archive [@doi:10.1038/s41592-018-0195-8]. 
If no specialist repository for that data type exists, one should instead use a generalist repository like Zenodo (https://zenodo.org) for datasets of up to 50 GB or Dryad (https://datadryad.org/) for datasets larger than 50GB. 
When researchers use existing datasets, they must include the code required to download and preprocess the data.

### Models
Sharing trained models is another critical component for reproducibility. 
Even if the code for an analysis were perfectly reproducible and required no extra scientist-time to run, its corresponding model would still need to be made publicly available. 
Requiring people who wish to use a method on their own data to re-train a model slows the progress of science, creates an unnecessary barrier to entry, and wastes the compute and effort of future researchers. 
Being unable to examine a model also makes trusting it difficult. 
Without access to the model it is hard to say whether the model fails to generalize to other datasets, fails to make fair decisions across demographic groups, or learns to make predictions based on artifacts in the data.

Because of the importance of sharing trained models, meeting the bronze standard of reproducibility requires that authors deposit trained weights for the models used to generate their results in a public repository. 
However, authors do not need to publish the weights for additional models from a hyperparameter sweep if one can reproduce the results without them. 
When a relevant specialist model zoo such as Kipoi [@doi:10.1038/s41587-019-0140-0] or Sfaira [@doi:10.1186/s13059-021-02452-6] exists, authors should deposit the models there. 
Otherwise, authors can deposit the models in a generalist repository such as Zenodo. 
Making models available solely on a non-archived website, such as a GitHub project, does not fulfill this requirement.

### Source Code
From a reproducibility standpoint, a work’s source code is as critical as its methods section. 
Source code contains implementation details that a future author is unlikely to replicate exactly from methods descriptions and reporting tables. 
These small deviations can lead to different behavior between the original work and the reproduced one. 
That is, of course, ignoring the huge burden of having to reimplement the entire analysis from scratch. 
For the computational components of a study, the code is likely a better description of the work than the methods section itself. 
As a result, computational papers without published code should meet similar skepticism to papers without methods sections.

To meet the bronze standard, authors must deposit code in a third-party, archivable repository like Zenodo. 
This includes the code used in training, tuning, and testing models, creating figures, processing data, and generating the final results. 
One good way of meeting the bronze standard involves creating a GitHub project and archiving it in Zenodo. 
Doing so gives both the persistence of Zenodo required by scholarly literature and GitHub’s resources for further development and use, such as the user support forum provided by GitHub Issues.

## Silver
While it is possible to reproduce an analysis with only its data, models, and code, this task is by no means easy. 
Fortunately there are best practices from the field of software engineering that can make reproducing analyses easier by simplifying package management, recording analysis details, and controlling randomness.

One roadblock that appears when attempting to reproduce an analysis stems from differences in behavior between versions of packages used in the analysis. 
Analyses that once worked with specific dependency versions can stop working altogether with later versions. 
Guessing which versions one must use to reproduce an analysis—or even to get it to run at all—can feel like playing a game of “package Battleship”. 
Proper use of dependency management tools like Packrat (https://rstudio.github.io/packrat/) and Conda (https://conda.io/) can eliminate these difficulties both for the authors and others seeking to build on the work by tracking which versions of packages are used.

Authors may also wish to consider containerization for managing dependencies. 
Container systems like Docker [@docker] allow authors to specify the system state in which to run their code more precisely than just versions of key software packages. 
Containerization provides better guarantees of reproducing a precise software environment, but this very fact can also facilitate code that won’t tolerate even modest environment changes. 
That brittleness can make it more difficult for future researchers to build on the original analysis. 
Therefore, we recommend that authors using containers also ensure that their code works on the latest version of at least one operating system distribution. 
Furthermore, containers do not fully insulate the running environment from the underlying hardware. 
Authors expecting bit-for-bit reproducibility from their containers may find that GPU-accelerated code fails to yield identical results on other machines due to the presence of different hardware or drivers.

Knowing the steps to run an analysis is a crucial part of reproducing it, yet this knowledge is often not formally recorded. 
It takes far less time for the original authors to document factors such as the order of analysis components or information about the computers used than for a third-party analyst attempting to reproduce the work to determine that information on their own. 
Accordingly, the silver standard requires that authors record the order in which one should run their analysis components, the operating system version used to produce the work, and the time taken to run the code. 
Authors must also list the system resources that yielded that time, such as the model and number of CPUs and GPUs and the amount of CPU RAM and GPU RAM required. 
Authors may record the order in which one should run components (1) in a README file within the code repository, (2) by adding numbers to the beginning of each script’s name to denote their order of execution, or (3) by providing a script to run them in order. 
Authors must include details on the operating system, wall clock and CPU running time, and system resources used both within the body of the manuscript and in the README.

The last challenge of this section, randomness, is common in machine learning analyses. 
Dataset splitting, neural network initialization, and even some GPU-parallelized math used in model training all include elements of randomness. 
Because models’ outputs depend heavily on these factors, the pseudorandom number generators used in analyses must be seeded to ensure consistent results. 
How the seeds are set depends on the language, though authors need to take special care when working with deep learning libraries. 
Current implementations often do not prioritize determinism, especially when accelerating operations on GPUs. 
However, some frameworks have options to mitigate nondeterministic operation (https://pytorch.org/docs/1.8.1/notes/randomness), and future versions may have fully deterministic operation (https://github.com/NVIDIA/framework-determinism). 
For now, the best way to account for this type of randomness is by publishing trained models. 
This nondeterminism is another reason why the minimal standard requires model publication—reproducing the model using data and code alone may prove impossible. 

As it is difficult to evaluate the extent to which an analysis follows best practices, we provide three requirements that must be met to achieve the silver standard in reproducibility. 
First, future users must be able to download and install all software dependencies for the analysis with a single command. 
Second, the order in which the analysis scripts should be run and how to run them should be documented. 
Finally, any random elements within the analysis should be made deterministic. 

## Gold
The gold standard for reproducibility requires the entire analysis to be reproducible with a single command. 
Achieving this goal requires authors to automate all the steps of their analysis, including downloading data, preprocessing data, training models, producing output tables, and generating and annotating figures. 
Full automation stands in addition to tracking dependencies and making their data and code available. 
In short, by meeting the gold standard authors make the burden of reproducing their work as small as possible.

Workflow management software such as Snakemake [@doi:10.1093/bioinformatics/bts480] or Nextflow [@doi:10.1038/nbt.3820] streamline the work of meeting the gold standard. 
They enable authors to create a series of rules that run all the components in an analysis. 
While a simple shell script can also accomplish this goal, workflow management software provides a number of advantages without extra work from the authors. 
For example, workflow management software can make it easy to restart analyses after errors, parallelize analyses, and track the progress of an analysis as it runs.

## Caveats
### Privacy
Not all data can be publicly released. 
Some data contain personally identifiable information or are restricted by a data use agreement. 
In these cases data should be stored in a controlled access repository [@doi:10.1038/s41576-020-0257-5], but the use of controlled access should be explicitly approved by journals to prevent it from becoming another form of “data available upon request”.

Training models on private data also poses privacy challenges. 
Models trained with standard workflows can be attacked to extract training data [@arxiv:2012.07805]. 
Fortunately, model training methods designed to preserve privacy exist: techniques such as differential privacy [@doi:10.1145/2976749.2978318] can help make models resistant to attacks seeking to uncover personally identifiable information, and can be applied with open source libraries such as Opacus (https://opacus.ai/). 
Researchers working on data with privacy constraints should employ these techniques as a routine practice.

When data cannot be shared, models must be shared to have any hope of computational reproducibility. 
If neither data nor models are published, the code is nearly useless, as it does not have anything to operate on. 
Future authors could perhaps replicate the study by recollecting data and regenerating the models, but they will not be able to evaluate the original analysis based on the published materials. 
When working on data with privacy restrictions, it is important for authors to use privacy preserving techniques for model training so that model release is not impeded. 
Studies with only models published will not be able to be fully reproduced, but there will at least be the possibility of testing the models’ behavior on other datasets.

### Compute-intensive analyses
Analyses can take a long time to run. 
In some cases they may take so long to run that it is infeasible for them to be reproduced by a different research group. 
In those cases, authors should store and publish intermediate outputs. 
Doing so allows other users to verify the final results even if they can not reproduce the entire pipeline. 
Workflow management systems, as mentioned in the gold standard section, make this partial reproduction straightforward by tracking intermediate outputs and using them to reproduce the final results automatically. 
Setting up a lightweight analysis demonstration, such as a web app on a small dataset or a Colab notebook (https://research.google.com/colaboratory/) running a pretrained model, can also be helpful for giving users the ability to evaluate model behavior without using large amounts of compute.

### Reproducibility of packages, libraries, and software products
The standards outlined in this paper focus on the computational reproducibility of analyses using machine learning. 
Standards for software designed for reuse, such as software packages and utilities, would have a broader scope and encompass more topics. 
In addition to our standards, such software should make use of unit testing, follow code style guidelines, have clear documentation [@doi:10.1093/bib/bbw134], and ensure compatibility across major operating systems to meet the gold standard for this type of research product.

## Conclusion
If we are to make machine learning research in the life sciences trustworthy, then we must make it computationally reproducible. 
Authors who strive to meet the bronze, silver, and gold standards will increase the reproducibility of machine learning analyses in the life sciences. 
These standards can also accelerate research in the field. In the status quo, there is no explicit reward for reproducible programming practices. 
As a result, authors can ostensibly minimize their own programming effort by using irreproducible programming practices and leaving future authors to make up the difference. 
In practice, irreproducible programming practices tend to decrease short-run effort for the authors, but increase effort in the long run on both the parts of the original authors and future reproducing authors. 
Implementing the standards in a way that rewards reproducible science (Box 1) helps avoid these long-run costs.

Ultimately, reproducibility in computational research is comparatively easy to experimental life science research. 
Computers are designed to perform the same tasks repeatedly with identical results. 
If we can not make purely computational analysis reproducible, how can we ever manage to make truly reproducible work in wet lab research with such variable factors as reagents, cell lines, and environmental conditions? 
If we want life science to lead the way in trustworthy, verifiable research, then setting standards for computational reproducibility is a good place to start.

### Acknowledgements
This work was supported by the Natural Sciences and Engineering Research Council of Canada (RGPIN-2015-03948 to M.M.H.); Cancer Research UK (A19274 to F.M.); and the National Institutes of Health’s National Institute of General Medical Sciences (R35 GM128638 to S.L.), the National Human Genome Research Institute (R00HG009007 to S.C.H. and R01HG010067 to C.S.G), and the National Cancer Institute of the National Institutes of Health (R01CA237170 to C.S.G.)

### Author contributions
Conceptualization, C.S.G. Project administration, B.J.H. Writing — original draft, B.J.H., S.C.H. Writing — review & editing, B.J.H., S.C.H., M.M.H., S.L., F.M., C.S.G.

### Ethics declarations
Competing interests: M.M.H. received an Nvidia GPU Grant.

## Box 1
**Journals**
Journals can enforce reproducibility standards as a condition of publication. 
The bronze standard should be the minimal standard, though some journals may wish to differentiate themselves by setting higher standards. 
Such journals may require the silver or gold standards for all manuscripts, or for particular classes of articles such as those focused on analysis. 
If journals act as the enforcing body for reproducibility standards, they can verify that the standards are met by either requiring reviewers to report which standards the work meets or by including a special reproducibility reviewer to evaluate the work.

**Badging**
A badge system that indicates the trustworthiness of work could incentivize scientists to progress to higher standards of reproducibility. 
Upon completing analyses, authors could submit their work to a badging organization that would then verify which standards of reproducibility their work met and assign a badge accordingly. 
Such an organization would likely operate in a similar way to the Bioconductor [@bioconductor] package review process. 
Authors could then include the badge with a publication or preprint to tout the effort the authors put in to ensure their code was reproducible. 
Including these badges in biosketches or CVs would make it simple to demonstrate a researcher’s track record of achieving high levels of reproducibility. 
This would provide a powerful signal to funding agencies and their reviewers that a researcher’s strengths in reproducibility would maximize the results of the investment made in a project. 
Universities could also promote reproducibility by explicitly requiring a track record of reproducible research in faculty hiring, annual review, and promotion.

**Reproducibility Collaborators**
Adding “reproducibility collaborators” to manuscripts would also provide another means to make analyses more reproducible. We envision a reproducibility collaborator as someone outside the primary authors’ research groups who certifies that they were able to reproduce the results of the paper from only the data, models, code, and accompanying documentation. Such collaborators would currently fall under the “validation” role in the CRediT Taxonomy (https://casrai.org/credit/), though it should be made clear that the reproducibility coauthor should not also be collaborating on the design or implementation of the analysis.

## Table 1
TODO COPY OVER Table 1 WHEN CONVERTING TO WORD


# Chapter 3: The Effect of Non-Linear Signal in Classification Problems using Gene Expression

This chapter has been preprinted at bioRxiv (https://www.biorxiv.org/content/10.1101/2022.06.22.497194v2), reviewed through Review Commons, and submitted for publication at PLOS Computational Biology as "The Effects of Nonlinear Signal on Expression-Based Prediction Performance" by Benjamin J. Heil, Jake Crawford, and Casey S. Greene.

**Contributions:**
I designed and ran the experiments, created the figures, and wrote/edited the manuscript.
Jake Crawford acted as the primary code reviewer, gave feedback and guidance on experiments, and edited the manuscript.
Casey S. Greene gave feedback and guidance on experiments and edited the manuscript.


## Abstract {.page_break_before}

Those building predictive models from transcriptomic data are faced with two conflicting perspectives.
The first, based on the inherent high dimensionality of biological systems, supposes that complex non-linear models such as neural networks will better match complex biological systems.
The second, imagining that complex systems will still be well predicted by simple dividing lines prefers linear models that are easier to interpret.
We compare multi-layer neural networks and logistic regression across multiple prediction tasks on GTEx and Recount3 datasets and find evidence in favor of both possibilities.
We verified the presence of non-linear signal when predicting tissue and metadata sex labels from expression data by removing the predictive linear signal with Limma, and showed the removal ablated the performance of linear methods but not non-linear ones.
However, we also found that the presence of non-linear signal was not necessarily sufficient for neural networks to outperform logistic regression.
Our results demonstrate that while multi-layer neural networks may be useful for making predictions from gene expression data, including a linear baseline model is critical because while biological systems are high-dimensional, effective dividing lines for predictive models may not be.


## Introduction

Transcriptomic data contains a wealth of information about biology.
Gene expression-based models are already being used for subtyping cancer [@doi:10.1200/JCO.2008.18.1370], predicting transplant rejections [@doi:10.1161/CIRCULATIONAHA.116.022907], and uncovering biases in public data [@pmc:PMC8011224].
In fact, both the capability of machine learning models [@arxiv:2202.05924] and the amount of transcriptomic data available [@doi:10.1038/s41467-018-03751-6; @doi:10.1093/database/baaa073] are increasing rapidly.
It makes sense, then, that neural networks are frequently being used to build predictive models from transcriptomic data [@doi:10.1038/s41598-019-52937-5; @doi:10.1093/gigascience/giab064; @doi:10.1371/journal.pcbi.1009433].

However, there are two conflicting ideas in the literature regarding the utility of non-linear models.
One theory draws on prior biological understanding: the paths linking gene expression to phenotypes are complex [@doi:10.1016/j.semcdb.2011.12.004; @doi:10.1371/journal.pone.0153295], and non-linear models like neural networks should be more capable of learning that complexity.
Unlike purely linear models such as logistic regression, non-linear models can learn non-linear decision boundaries to differentiate phenotypes.
Accordingly, many have used non-linear models to learn representations useful for making predictions of phenotypes from gene expression [@doi:10.1128/mSystems.00025-15; @doi:10.1016/j.cmpb.2018.10.004; @doi:10.1186/s12859-017-1984-2].

The other supposes that even high-dimensional complex systems may have linear decision boundaries.
This is supported empirically: linear models seem to do as well as or better than non-linear ones in many cases [@doi:10.1186/s12859-020-3427-8].
While papers of this sort are harder to come by — perhaps scientists do not tend to write papers about how their deep learning model was worse than logistic regression — other complex biological problems have also seen linear models prove equivalent to non-linear ones [@doi:10.1016/j.jclinepi.2019.02.004; @doi:10.1038/s41467-020-18037-z].

We design experiments to ablate linear signal and find merit to both hypotheses.
We construct a system of binary and multiclass classification problems on the GTEx and Recount3 compendia [@doi:10.1038/ng.2653;@doi:10.1186/s13059-021-02533-6] that shows linear and non-linear models have similar accuracy on several prediction tasks.
However, when we remove any linear separability from the data, we find non-linear models are still able to make useful predictions even when the linear models previously outperformed the non-linear ones.
Given the unexpected nature of these findings, we evaluate independent tasks, examine different problem formulations, and verify our models' behavior with simulated data.
The models' results are consistent across each setting, and the models themselves are comparable, as they use the same training and hyperparameter optimization processes [@pmcid:PMC6417816].

In reconciling these two ostensibly conflicting theories, we confirm the importance of implementing and optimizing a linear baseline model before deploying a complex non-linear approach.
While non-linear models may outperform simpler models at the limit of infinite data, they do not necessarily do so even when trained on the largest datasets publicly available today.


## Results

### Linear and non-linear models have similar performance in many tasks
We compared the performance of linear and non-linear models across multiple datasets and tasks (fig. @fig:workflow A).
We examined using TPM-normalized RNA-seq data to predict tissue labels from GTEx [@doi:10.1038/ng.2653], tissue labels from Recount3 [@doi:10.1186/s13059-021-02533-6], and metadata-derived sex labels from Flynn et al. [@doi:10.1186/s12859-021-04070-2].
To avoid leakage between cross-validation folds, we placed entire studies into single folds (fig. @fig:workflow B).
We evaluated models on subsampled datasets to determine the extent to which performance was affected by the amount of training data.

![
Schematic of the model analysis workflow. We evaluate three models on multiple classification problems in three datasets (A). We stratify the samples into cross-validation folds based on their study (in Recount3) or donor (in GTEx). We also evaluate the effects of sample-wise splitting and pretraining (B).
](./images/workflow.svg "Workflow diagram"){#fig:workflow width="100%"}


We used GTEx [@doi:10.1038/ng.2653] to determine whether linear and non-linear models performed similarly on a well-characterized dataset with consistent experimental protocols across samples.
We first trained our models to differentiate between tissue types on pairs of the five most common tissues in the dataset.
Likely due to the clean nature of the data, all models were able to perform perfectly on these binary classification tasks (fig. @fig:signal_removed_binary A).
Because binary classification was unable to differentiate between models, we evaluated the models on a more challenging task.
We tested the models on their ability to perform multiclass classification on all 31 tissues present in the dataset.
In the multitask setting, logistic regression slightly outperformed the five-layer neural network, which in turn slightly outperformed the three-layer net (fig. @fig:signal_removed_multiclass A).

We then evaluated the same approaches in a dataset with very different characteristics: Sequence Read Archive [@doi:10.1093/nar/gkq1019] samples from Recount3 [@doi:10.1186/s13059-021-02533-6].
We compared the models' ability to differentiate between pairs of tissues (supp. fig. @fig:signal_removed_binary B) and found their performance was roughly equivalent.
We also evaluated the models' performance on a multiclass classification problem differentiating between the 21 most common tissues in the dataset.
As in the GTEx setting, the logistic regression model outperformed the five-layer network, which outperformed the three-layer network (fig. @fig:signal_removed_multiclass B). 

To examine whether these results held in a problem domain other than tissue type prediction, we tested performance on metadata-derived sex labels (fig. @fig:signal_removed_multiclass C), a task previously studied by Flynn et al. [@doi:10.1186/s12859-021-04070-2].
We used the same experimental setup as in our other binary prediction tasks to train the models, but rather than using tissue labels we used sex labels from Flynn et al.
In this setting we found that while the models all performed similarly, the non-linear models tended to have a slight edge over the linear one.

![
Performance of models across three classification tasks before and after signal removal. 
In each panel the loess curve and its 95% confidence interval are plotted based on points from three seeds, ten data subsets, and five folds of study-wise cross-validation (for a total of 150 points per model per panel). 
It is worth noting that "Sample Count" in these figures refers to the total number of RNA-seq samples, some of which share donors. 
As a result, the effective sample size may be lower than the sample count.
](./images/signal_removed_multiclass.svg ){#fig:signal_removed_multiclass width="100%"}

### There is predictive non-linear signal in transcriptomic data
Our results to this point are consistent with a world where the predictive signal present in transcriptomic data is entirely linear.
If that were the case, non-linear models like neural networks would fail to give any substantial advantage.
However, based on past results we expect there to be relevant non-linear biological signal [@doi:10.1101/2022.06.15.496326].
To get a clearer idea of what that would look like, we simulated three datasets to better understand model performance for a variety of data generating processes.
We created data with both linear and non-linear signal by generating two types of features: half of the features with a linear decision boundary between the simulated classes and half with a non-linear decision boundary (see [Methods](#methods) for more details).
After training to classify the simulated dataset, all models effectively predicted the simulated classes.
To determine whether or not there was non-linear signal, we then used Limma [@doi:10.1093/nar/gkv007] to remove the linear signal associated with the endpoint being predicted.
After removing the linear signal from the dataset, non-linear models correctly predicted classes, but logistic regression performed no better than random (fig. @fig:simulation B).

To confirm that non-linear signal was key to the performance of non-linear methods, we generated another simulated dataset consisting solely of features with a linear decision boundary between the classes.
As before, all models were able to predict the different classes well.
However, once the linear signal was removed, all models performed no better than random guessing (fig. @fig:simulation A).
That the non-linear models only achieved baseline accuracy also indicated that the signal removal method was not injecting non-linear signal into data where non-linear signal did not exist.

We also trained the models on a dataset where all features were Gaussian noise as a negative control.
As expected, the models all performed at baseline accuracy both before and after the signal removal process (fig. @fig:simulation C).
This experiment supported our decision to perform signal removal on the training and validation sets separately.
One potential failure state when using the signal removal method would be if it induced new signal as it removed the old.
Such a state can be seen when removing the linear signal in the full dataset (supp. fig. @fig:split-signal-correction).

![
Performance of models in binary classification of simulated data before and after signal removal. Dotted lines indicate expected performance for a naive baseline classifier that predicts the most frequent class.
](./images/simulated_data_combined.svg ){#fig:simulation width="100%"}

We next removed linear signal from GTEx and Recount3.
We found that the neural nets performed better than the baseline while logistic regression did not (fig. @fig:signal_removed_multiclass,  fig. @fig:signal_removed_binary).
For multiclass problems logistic regression performed poorly while the non-linear models had performance that increased with an increase in data while remaining worse than before the linear signal was removed (fig. @fig:signal_removed_multiclass A, B)
Likewise, the sex label prediction task showed a marked difference between the neural networks and logistic regression: only the neural networks could learn from the data (fig. @fig:signal_removed_multiclass C).
In each of the settings, the models performed less well when run on data with signal removed, indicating an increase in the problem's difficulty. 
Logistic regression, in particular, performed no better than random.

![
Models' performance across binary classification tasks before and after signal removal in the Recount and GTEx datasets.
](./images/signal_removed_binary.svg ){#fig:signal_removed_binary width="100%"}

To verify that our results were not an artifact of our decision to assign studies to cross-validation folds rather than samples, we compared the study-wise splitting that we used with an alternate method called sample-wise splitting.
Sample-wise splitting (see [Methods](#methods)) is common in machine learning, but can leak information between the training and validation sets when samples are not independently and identically distributed among studies - a common feature of data in biology [@doi:10.1038/s41576-021-00434-9].
We found that sample-wise splitting induced substantial performance inflation (supp. fig. @fig:splitting).
The relative performance of each model stayed the same regardless of the data splitting technique, so the results observed were not dependent on the choice of splitting technique.

Another growing strategy in machine learning, especially on biological data where samples are limited, is training models on a general-purpose dataset and fine-tuning them on a dataset of interest.
We examined the performance of models with and without pretraining (supp. fig. @fig:pretrain).
We split the Recount3 data into three sets: pretraining, training, and validation (fig. @fig:workflow B), then trained two identically initialized copies of each model.
One was trained solely on the training data, while the other was trained on the pretraining data and fine-tuned on the training data.
The pretrained models showed high performance even when trained with small amounts of data from the training set.
However, the non-linear models did not have a greater performance gain from pretraining than logistic regression, and the balanced accuracy was similar across models.


## Methods

### Datasets

#### GTEx

We downloaded the 17,382 TPM-normalized samples of bulk RNA-seq expression data available from version 8 of GTEx.
We zero-one standardized the data and retained the 5000 most variable genes.
The tissue labels we used for the GTEx dataset were derived from the 'SMTS' column of the sample metadata file.

#### Recount3

We downloaded RNA-seq data from the Recount3 compendium [@pmc:PMC86284] during the week of March 14, 2022.
Before filtering, the dataset contained 317,258 samples, each containing 63,856 genes.
To filter out single-cell data, we removed all samples with greater than 75 percent sparsity.
We also removed all samples marked 'scrna-seq' by Recount3's pattern matching method (stored in the metadata as 'recount_pred.pattern.predict.type').
We then converted the data to transcripts per kilobase million using gene lengths from BioMart [@pmc:PMC2649164] and performed standardization to scale each gene's range from zero to one.
We kept the 5,000 most variable genes within the dataset.

We labeled samples with their corresponding tissues using the 'recount_pred.curated.tissue' field in the Recount3 metadata.
These labels were based on manual curation by the Recount3 authors.
A total of 20,324 samples in the dataset had corresponding tissue labels.
Samples were also labeled with their corresponding sex using labels from Flynn et al. [@pmc:PMC8011224].
These labels were derived using pattern matching on metadata from the European Nucleotide Archive [@pmc:PMC3013801].
A total of 23,525 samples in our dataset had sex labels.

#### Data simulation

We generated three simulated datasets.
The first dataset contained 1,000 samples of 5,000 features corresponding to two classes.
Of those features, 2,500 contained linear signal.
That is to say that the feature values corresponding to one class were drawn from a standard normal distribution, while the feature values corresponding to the other were drawn from a Gaussian with a mean of 6 and unit variance.

We generated the non-linear features similarly.
The values for the non-linear features were drawn from a standard normal distribution for one class, while the second class had values drawn from either a mean six or negative six Gaussian with equal probability.
These features are referred to as "non-linear" because two dividing lines are necessary to perfectly classify such data, while a linear classifier can only draw one such line per feature.

The second dataset was similar to the first dataset, but it consisted solely of 2,500 linear features.
The final dataset contained only values drawn from a standard normal distribution regardless of class label.

### Model architectures

We used three representative models to demonstrate the performance profiles of different model classes.
The first was a linear model, ridge logistic regression, selected as a simple linear baseline to compare the non-linear models against.
The next model was a three-layer fully-connected neural network with ReLU non-linearities [@https://dl.acm.org/doi/10.5555/3104322.3104425] and hidden layers of size 2500 and 1250.
This network served as a model of intermediate complexity: it was capable of learning non-linear decision boundaries, but not the more complex representations a deeper model might learn.
Finally, we built a five-layer neural network to serve as a (somewhat) deep neural net.
This model also used ReLU non-linearities, and had hidden layers of sizes 2500, 2500, 2500, and 1250.
The five-layer network, while not particularly deep compared to, e.g., state of the art computer vision models, was still in the domain where more complex representations could be learned, and vanishing gradients had to be accounted for.

### Model training

We trained our models via a maximum of 50 epochs of mini-batch stochastic gradient descent in PyTorch [@arxiv:1912.01703].
Our models minimized the cross-entropy loss using an Adam [@arxiv:1412.6980] optimizer.
They also used inverse frequency weighting to avoid giving more weight to more common classes.
To regularize the models, we used early stopping and gradient clipping during the training process.
The only training differences between the models were that the two neural nets used dropout [@https://jmlr.org/papers/v15/srivastava14a.html] with a probability of 0.5, and the deeper network used batch normalization [@https://proceedings.mlr.press/v37/ioffe15.html] to mitigate the vanishing gradient problem.

We ensured the results were deterministic by setting the Python, NumPy, and PyTorch random seeds for each run, as well as setting the PyTorch backends to deterministic and disabling the benchmark mode.
The learning rate and weight decay hyperparameters for each model were selected via nested cross-validation over the training folds at runtime, and we tracked and recorded our model training progress using Neptune [@neptune].

We also used Limma[@doi:10.1093/nar/gkv007] to remove linear signal associated with tissues in the data.
We ran the 'removeBatchEffect' function on the training and validation sets separately, using the tissue labels as batch labels.
This function fits a linear model that learns to predict the training data from the batch labels, and uses that model to regress out the linear signal within the training data that is predictive of the batch labels.

### Model Evaluation
In our analyses we used five-fold cross-validation with study-wise data splitting.
In a study-wise split, the studies are randomly assigned to cross-validation folds such that all samples in a given study end up in a single fold (fig. @fig:workflow B).

**Hardware**  
Our analyses were performed on an Ubuntu 18.04 machine and the Colorado Summit compute cluster.
The desktop CPU used was an AMD Ryzen 7 3800xt processor with 16 cores and access to 64 GB of RAM, and the desktop GPU used was an Nvidia RTX 3090.
The Summit cluster used Intel Xeon E5-2680 CPUs and NVidia Tesla K80 GPUs.
From initiating data download to finishing all analyses and generating all figures, the full Snakemake [@doi:10.1093/bioinformatics/bts480] pipeline took around one month to run.

**Recount3 tissue prediction**  
In the Recount3 setting, the multi-tissue classification analyses were trained on the 21 tissues (see Supp. Methods) that had at least ten studies in the dataset.
Each model was trained to determine which of the 21 tissues a given expression sample corresponded to.

To address class imbalance, our models' performance was then measured based on the balanced accuracy across all classes.
Unlike raw accuracy, balanced accuracy (the mean across all classes of the per-class recall) isn't predominantly determined by performance on the largest class in an imbalanced class setting.
For example, in a binary classification setting with 9 instances of class A and 1 instance of class B, successfully predicting 8 of the 9 instances of class A and none of class B yields an accuracy of 0.8 and a balanced accuracy of 0.44.

The binary classification setting was similar to the multiclass one.
The five tissues with the most studies (brain, blood, breast, stem cell, and cervix) were compared against each other pairwise.
The expression used in this setting was the set of samples labeled as one of the two tissues being compared.

The data for both settings were split in a stratified manner based on their study.

**GTEx classification**  
The multi-tissue classification analysis for GTEx used all 31 tissues.
The multiclass and binary settings were formulated and evaluated in the same way as in the Recount3 data.
However, rather than being split study-wise, the cross-validation splits were stratified according to the samples' donors.

**Simulated data classification/sex prediction**  
The sex prediction and simulated data classification tasks were solely binary.
Both settings used balanced accuracy, as in the Recount3 and GTEx problems.

**Pretraining**  
When testing the effects of pretraining on the different model types, we split the data into three sets.
Approximately forty percent of the data went into the pretraining set, forty percent went into the training set, and twenty percent went into the validation set.
The data was split such that each study's samples were in only one of the three sets to simulate the real-world scenario where a model is trained on publicly available data and then fine-tuned on a dataset of interest.

To ensure the results were comparable, we made two copies of each model with the same weight initialization.
The first copy was trained solely on the training data, while the second was trained on the pretraining data, then the training data.
Both models were then evaluated on the validation set.
This process was repeated four more times with different studies assigned to the pretraining, training, and validation sets.


## Discussion and Conclusion

We performed a series of analyses to determine the relative performance of linear and non-linear models across multiple tasks.
Consistent with previous papers [@doi:10.1186/s12859-020-3427-8; @doi:10.1016/j.jclinepi.2019.02.004], linear and non-linear models performed roughly equivalently in a number of tasks.
That is to say that there are some tasks where linear models perform better, some tasks where non-linear models have better performance, and some tasks where both model types are equivalent.

However, when we removed all linear signal in the data, we found that residual non-linear signal remained.
This was true in simulated data as well as GTEx and Recount3 data across several tasks.
These results also held in altered problem settings, such as using a pretraining dataset before the training dataset and using sample-wise data splitting instead of study-wise splitting.
This consistent presence of non-linear signal demonstrated that the similarity in performance across model types was not due to our problem domains having solely linear signals.

One limitation of our study is that the results likely do not hold in an infinite data setting.
Deep learning models have been shown to solve complex problems in biology and tend to significantly outperform linear models when given enough data.
However, we do not yet live in a world in which millions of well-annotated examples are available in many areas of biology.
Our results are generated on some of the largest labeled expression datasets in existence (Recount3 and GTEx), but our tens of thousands of samples are far from the millions or billions used in deep learning research.

We are also unable to make claims about all problem domains or model classes.
There are many potential transcriptomic prediction tasks and many datasets to perform them on.
While we show that non-linear signal is not always helpful in tissue or sex prediction, and others have shown the same for various disease prediction tasks, there may be problems where non-linear signal is more important.
It is also possible that other classes of models, be they simpler non-linear models or different neural network topologies, are more capable of taking advantage of the non-linear signal present in the data.

Ultimately, our results show that task-relevant non-linear signal in the data, which we confirm is present, does not necessarily lead non-linear models to outperform linear ones.
Additionally, our results suggest that scientists making predictions from expression data should always include simple linear models as a baseline to determine whether more complex models are warranted.


### Code and Data Availability
The code, data, and model weights to reproduce this work can be found at https://github.com/greenelab/linear_signal.
Our work meets the bronze standard of reproducibility [@doi:10.1038/s41592-021-01256-7] and fulfills aspects of the silver and gold standards including deterministic operation and an automated analysis pipeline.

### Acknowledgements
We would like to thank Alexandra Lee and Jake Crawford for reviewing code that went into this project.
We would also like to thank the past and present members of GreeneLab who gave feedback on this project during lab meetings.
This work utilized resources from the University of Colorado Boulder Research Computing Group, which is supported by the National Science Foundation (awards ACI-1532235 and ACI-1532236), the University of Colorado Boulder, and Colorado State University.

#### Funding
This work was supported by grants from the National Institutes of Health’s National Human Genome Research Institute (NHGRI) under award R01 HG010067 and the Gordon and Betty Moore Foundation (GBMF 4552) to CSG. 
The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.


# Supplementary Materials
## Results 

### Recount binary classification 

### Signal removal
While it's possible to remove signal in the full dataset or the train and validation sets independently, we decided to do the latter.
We made this decision because we observed potential data leakage when removing signal from the entire dataset in one go (supp. fig. @fig:split-signal-correction).

![                                                                                                                                                                                                          
Full dataset signal removal in a dataset without signal
](./images/no_signal_sim_signal_removed.svg "Signal removal from data with no signal to begin with"){#fig:split-signal-correction} 

![                                                                                                                                                                                                          
Comparison of models' binary classification performance before and after removing linear signal                                                                                                             
](./images/recount_binary_combined.svg "Recount binary classification before and after signal removal"){#fig:recount-binary-combined}


### Samplewise splitting
![                                                                                                                                                                                                          
Performance of Recount3 multiclass prediction with samplewise train/val splitting
](./images/recount_multiclass_sample_split.svg ){#fig:splitting}

### Recount3 Pretraining 
![                                                                                                                                                                                                          
Performance of Recount3 multiclass prediction with pretraining
](./images/recount_pretraining.svg "Pretraining"){#fig:pretrain} 

## Methods
### Recount3 tissues used
The tissues used from Recount3 were blood, breast, stem cell, cervix, brain, kidney, umbilical cord, lung, epithelium, prostate, liver, heart, skin, colon, bone marrow, muscle, tonsil, blood vessel, spinal cord, testis, and placenta.



# Chapter 4: MousiPLIER, the largest and most murine PLIER model ever trained

This chapter is from a manuscript in progress that is a collaboration between the Heller and Greene labs.

**Contributions**  
I am co-first author on the manuscript. 
I trained the MousiPLIER model, wrote and edited the manuscript, wrote code to make the results easier to use in Python, and ran the analyses leading to figures 9, 10, 12, and 15.
Shuo Zhang is the other co-first author on the manuscript.
He edited the manuscript, provided biological expertise useful in selecting latent variables and gene sets, and ran the analyses leading to figures 11, 13, and 14.
Wayne Mao provided guidance for training PLIER in a large dataset.
Casey S. Greene gave feedback and guidance on the experiments run, and is a co-corresponding author on the manuscript.
Elizabeth A. Heller edited the manuscript, gave feedback and guidance on the experiments run, and helped interpret enriched genes in MousiPLIER latent variables.


## Abstract {.page_break_before}

Differential expression analysis is widely used to learn from gene expression data.
However, it suffers from the curse of dimensionality — RNA-sequencing experiments tends to have tens of thousands of genes with only tens or hundreds of samples.
Many unsupervised learning models are designed to reduce dimensionality, and the PLIER model in particular fits expression data well.
In this paper we describe the training of the Mouse MultiPLIER (MousiPLIER) model, the first PLIER model trained on a mouse compendium and the PLIER model with the most training samples.
We then go on to show that the model's latent variables contain biologically relevant information by finding enrichment for a striatally-associated latent variable in a mouse brain aging study and using the latent variable to uncover studies in the training data corresponding to mouse brain processes.
This new model can assist mouse researchers in understanding the biological processes involved in their study and finding other studies in which these processes are relevant.


## Introduction

A common way to gain knowledge from gene expression is by examining which genes are differentially expressed in an experiment [@doi:10/dcz39r; @doi:10.1186/gb-2010-11-10-r106; @doi:10.1371/journal.pone.0190152; @doi:10.1093/bioinformatics/btab327].
For example, one can find which genes' expression values change in mouse tissue samples before and after some biological perturbation.
This approach can be challenging, however, as studies analyzing gene expression tend to have few samples compared to the number of genes.
The resulting lack of statistical power can be addressed by increasing the number of samples in an experiment, which is expensive.
Alternatively, one can reduce the dimensionality or use information from outside the study.

Unsupervised learning does both.
It is a category of methods from the field of machine learning that learn the structure of data without need for any biological labels denoting which experimental conditions are present.
Such methods are well-suited for gene expression data, and are often used for tasks such as reducing the dimensionality of expression datasets [@doi:10.1037/h0071325; @tsne; @doi:10.21105/joss.00861], clustering samples [@doi:10.4137/BBI.S38316; @doi:10.1093/bioinformatics/btz769], or learning shared expression patterns across experiments [@doi:10.1128/mSystems.00025-15; @doi:10.1093/bioinformatics/btz338].
That being said, while unsupervised models are capable of using large amounts of unlabeled expression data, many of them don't explicitly encode prior biological knowledge to encourage the model to learn biological patterns over technical ones.

The Pathway-level information extractor (PLIER) models do [@doi:10.1038/s41592-019-0456-1].
They are built explicitly to work on expression data, and use matrix factorization to incorporate prior knowledge in the form of sets of genes specified by the user corresponding to biological pathways or cell type markers.
PLIER models also capable of learning diverse biological pathways from entire compendia of expression data and transferring that knowledge to smaller studies as seen in MultiPLIER [@doi:10.1016/j.cels.2019.04.003].
However, PLIER models are largely trained on a single dataset rather than a compendium [@doi:10.1038/s41598-019-57110-6; @doi:10.1038/s41586-022-05056-7; @doi:10.1016/j.celrep.2022.110467], and past MultiPLIER runs have only trained models with up to tens of thousands of samples [@doi:10.3390/genes11020226; @doi:10.1016/j.cels.2019.04.003].

In this paper we train a PLIER model on a compendium of mouse gene expression to convert the data into a series of values called "latent variables" that correspond to potentially biologically relevant combinations of genes.
In doing so we train the largest (in terms of training data) PLIER model to date and the first one trained on a mouse compendium.
We have named this model MousiPLIER, short for Mouse MultiPLIER.
We demonstrate that not only is training such a model possible, it also surfaces interesting biology in a study of mouse brain aging.
When looking at a novel view of the resulting data (k-means clusters in the latent variable space), we find that the microglia-associated latent variables from our study of interest also correspond to aging-related changes in the training data.
Finally, we build a web server to allow others to visualize the results and find patterns in the data based on their own latent variables of interest.
Going forward, this model and its associated web server will be a useful tool for better understanding mouse gene expression.


## Results

### MousiPLIER learns latent variables with ideal pathway level and gene-level sparsity

To determine the utility of a mouse multiPLIER model, we first trained the model.
We then examined the latent variables that our model learned and found which were significantly enriched in a mouse brain aging study.
Next, we looked deeper into the significant wild-type microglia-associated latent variables and determined that they were learning striatal signals potentially relevant to aging.
Ultimately, we found that our trained PLIER model is able to uncover relevant signal in individual studies, and will be useful going forward in uncovering the biological processes present in mouse transcriptomic experiments.
Taken together this study provides proof of concept that the mouse PLIER model can surface meaningful biological processes in mouse transcriptomic studies.

We trained MousiPLIER by using an on-disk PCA implementation to initialize PLIER, modifying the pipeline to work with mouse data, and using a high-mem compute node to manage the size of the matrix decomposition (see Methods for more details).
The resulting model had 196 latent variables, where the per-latent variable distribution had an average of around 65% sparsity, which is to say that the latent variables tended to use only around 35% of the genes in the training data (Fig. @fig:genesparsity).
While many of the latent variables corresponded to no pathways, indicating signals in the training data not passed in as prior knowledge, those that remained corresponded to few pathways (Fig. @fig:pathwaysparsity).
This pathway-level and gene-level sparsity is ideal, as it allows us to interrogate individual latent variables that correspond to a small number of biological functions.

![
The distribution of the percentage of genes from the training set which had nonzero loadings for the latent variable.
](./images/filtered_percent_genes_used_hist.png "Sparsity per latent variable."){#fig:genesparsity width="75%"}

![
The distribution of the number of prior knowledge gene sets used per latent variable.
](./images/filtered_lv_per_pathway_hist.png "Genesets used per latent variable"){#fig:pathwaysparsity width="75%"}

### Some latent variables in MousiPLIER are enriched in wild-type microglia
With our model trained, we began interrogating our latent variables.
Because we were interested in brain-relevant latent variables that our model had learned from the compendium, we analyzed a study on mouse brain aging from Pan et al. [@doi:10.1186/s12974-020-01774-9].
In this study, microglia and astrocytes from five ages of mice were sequenced to see how their gene expression changed over time.
To determine which (if any) latent variables were changed across developmental aging in the study, we used a linear model to find the latent variables that changed significantly as the cells aged.
We found that each condition in the study had a set of significant latent variables, but that they were largely disjoint (Fig. @fig:venn).
To narrow down the scope of the analysis, we decided to validate the biological relevance of the latent variables associated with wild-type microglial cells.

![
The overlap in significantly enriched latent variables across types and experimental conditions.
](./images/dif_LV_venn.png "Latent variable overlap"){#fig:venn width="100%"}

### Latent variable 41 demonstrates the biological relevance of mousiplier latent variables
Once we had microglia-associated latent variables of interest, we set out to find which experiments in the training data responded strongly to them.
To do so, we developed a novel method of ranking experiments based on their latent variable weights.
More precisely, we performed *k*-means clustering with a *k* of two on each experiment in each latent variable space, and ranked experiments by their silhouette scores.
This procedure allowed us to uncover experiments where there were two groups of samples with distinct sets of values for our latent variables of interest.

We focused this approach on latent variable 41, which contains genes functionally associated with striatal cell type specificity. 
Upon examining the top ranked studies, we found that several with high silhouette scores for latent variable 41 corresponded to processes ocurring in the brain (Fig. @fig:clusters).
We dug deeper into which samples in particular were present in each clustered experiment and found our latent variable was in fact reflecting a biological process ocurring in the striatum and cortex but not the cerebellum or non-brain tissues (Fig. @fig:brain).
For example, in study SRP070440 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76872) the striatal samples clearly stand apart from the other neuronal tissues.
Similarly, we found that the two distinct groups in SRP047452 [@doi:10.1073/pnas.1411263112] were made up of brain samples from embryonic and adult mice, supporting the association between latent variable 41 and aging we found in the study we used to derive the latent variables (Fig. @fig:aging).

![
Latent variable 41 expression values are higher for striatum tissue samples than other tissues in SRP070440.
](./images/lv_41_studies.svg "Study clusters"){#fig:clusters width="100%"}

![ 
Striatal values for latent variable 41 compared to other tissues and brain regions.
](./images/brain_samples.png "Striatum LV41"){#fig:brain width="100%"}

![
The effects of aging on latent variable 41.
](./images/aging_samples.png "LV41 aging"){#fig:aging width="100%"}

To allow others to look at the learned latent variables on their own, we have set up a web server at https://www.mousiplier.greenelab.com. 
This server allows users to list the genes present in, visualize which experiments had high cluster scores for, and see which biological pathways participate in each latent variable (Fig. @fig:webserver).

![
An image of the webserver displaying the per-latent variable experiment ranking feature.
](./images/webserver.png "webserver"){#fig:webserver width="100%"}


## Methods

### Data
We began by downloading all the mouse gene expression data in Recount3, along with its corresponding metadata [@doi:10.1186/s13059-021-02533-6].
We then removed the single-cell RNAseq data from the dataset to ensure our data sources were fairly consistent across samples and studies.
Next, we filtered the expression data, keeping only genes that overlapped between Recount3 and our prior-knowledge genesets.
Finally, we RPKM transformed the expression using gene lengths from the Ensembl BioMart database [@doi:10.1093/nar/gkaa942] and Z-scored the expression data to ensure a consistent range for the downstream PLIER model.

For our prior knowledge gene sets we used cell type marker genes from CellMarker [@doi:10.1093/nar/gky900], pathway gene sets from Reactome [@doi:10.1093/nar/gkab1028], and manually curated brain marker genes.
We selected cell type marker genes corresponding to all available mouse cell types within the CellMarker database.
For mouse biological pathways, we downloaded pathway information from the Reactome database.
More specifically, we processed the files "Ensembl2Reactome_All_Levels.txt", "ReactomePathways.txt", and "ReactomePathwaysRelation.txt", selecting only pathways using mouse genes, filtering out all pathways with fewer than 5 genes present, and keeping only pathways that were leaf nodes on the pathway hierarchy.
Because we were interested in mouse brains in particular, we rounded out our set of prior information by manually selecting marker genes for the striatum, midbrain, and cerebellum.
In total, we used 1,003 prior knowledge pathways when training our model.

### PLIER 
We began the PLIER pipeline by precomputing the initialization for PLIER with incremental PCA in scikit-learn [@skl].
We then used the expression compendium, prior knowledge genesets, and PCA initializations to train a PLIER model.
The resulting task took two days to run and yielded 196 latent variables.

### Latent variable significance
To determine which latent variables were associated with experimental conditions, we used a linear model.
To correct the p-values for multiple testing, we used the Benjamani-Hochberg procedure [@doi:10.1111/j.2517-6161.1995.tb02031.x].

### Clustering 
We selected the latent variables significantly associated with aging in mouse microglia as a biological starting point.
We then used these latent variables to query the training data and see which studies seemed associated with the same biological signals.
To do so, we used *k*-means clustering with a *k* of 2, to look for experiments where there was some experimental condition that affected the latent variable.
We then ranked the top ten studies based on their silhouette scores, and looked to see which conditions were associated with relevant experimental variables.

### Hardware
The PLIER model training was performed on the Penn Medicine high performance computing cluster.
The full pipeline takes around two weeks to run, with the main bottlenecks being the Recount3 data download, which takes one week to run, and training the PLIER model, which takes two days on a compute node with 250GB of RAM.

### Web Server
The web server for visualizing the results was built on top of the ADAGE web app framework [@doi:10.1186/s12859-017-1905-4].
The main changes we made were to substitute the latent variables and gene sets from our trained PLIER model, to use clusters' silhouette scores for ranking experiments, and to forgo uploading the input expression data as the mouse compendium we used was much larger than the input expression for ADAGE.


## Discussion/Conclusion

In this paper we demonstrate that it is possible to train large PLIER models on mouse data.
We then show that the learned latent variables map to various biological processes and cell types.
We also describe a novel approach for surfacing latent-variable relevant experiments from an expression compendium. 
Namely, we cluster them based on latent variable values, allowing us to query a large compendium for experiments pertaining to mouse striatal aging.
Finally, we create a web server to make the model's results more easily accessible to other scientists.

Our study is not without its limitations though.
We show how a study from outside the training data can be transformed into the latent space to see which of the learned latent variables have significant differences in the study.
However, not all studies will have significant changes in latent variables between their experimental conditions.
This may be due to lack of similar samples in training compendium, too few samples in the study of interest, or other factors.
In these cases, there isn't a good way to select which latent variables should be used for downstream analyses.

Additionally, PLIER is a linear model.
If there are non-linear relationships between the genes used to train the model and the learned biological pathways, at best PLIER can approximate them.
While we do not expect this to have a large impact [@doi:10.1101/2022.06.22.497194], incorporating prior knowledge into non-linear models such as neural networks is an exciting field of research and a potential improvement for the MultiPLIER framework we use.

Going forward, our model and web server will allow scientists to explore the latent space of their own experiments and learn about relevant biological pathways and cell types.


### Code and Data Availability
The code, and model weights to reproduce this work can be found at https://github.com/greenelab/mousiplier.
The data used in our analyses is publicly available and can be downloaded with the code above or is already stored in the repository.
Our work meets the bronze standard of reproducibility [@doi:10.1038/s41592-021-01256-7].

### Acknowledgements
We would like to thank Jake Crawford for reviewinging code that went into this project.
We would also like to thank Faisal Alquaddoomi and Vincent Rubinetti for their assistance in developing the web server accompanying this project.
This work utilized resources from the the University of Pennsylvania PMACS/DART computer cluster funded by NIH grant 1S10OD012312.

#### Funding
This work was supported by grants from the National Institutes of Health’s National Human Genome Research Institute (NHGRI) under award R01 HG010067 and the Gordon and Betty Moore Foundation (GBMF 4552) to CSG.
The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript. 


# Chapter 5 - The Field-Dependent Nature of PageRank Values in Citation Networks

**Contributions:**  
I ran the experiments, generated the figures, and wrote the manuscript for this chapter.
Casey S. Greene gave advice and feedback, helped design experiments, and edited the manuscript.


## Abstract {.page_break_before}

There are more academic papers than any human can read in a lifetime, so several article-level and journal-level metrics have been devised to rank papers.
One challenge when creating such metrics is the differences in citation practices between fields.
To account for these differences, scientists have devised normalization schemes to make metrics more comparable across fields.
In this paper, we argue that these normalization schemes obscure useful signals about fields' preferences for articles.
We use PageRank as an example metric and begin by demonstrating that there are, in fact, differences in journals' PageRanks between fields.
We then show that even papers shared between fields have different PageRanks depending on which field's citation network the metric is calculated in.
Finally, we find that some of these differences are caused by field-specific preferences by using a degree-preserving graph shuffling algorithm to generate a null distribution of similar networks.
Our results demonstrate that while differences exist between fields' metric distributions, applying metrics in a field-aware manner rather than using normalized global metrics avoids losing important information about article preferences. 


## Introduction

There are more academic papers than any human can read in a lifetime.
Attention has been given to ranking papers, journals, or researchers by their "importance," assessed via various metrics.
Citation count assumes the number of citations determines a paper's importance.
The h-index and Journal Impact Factor focus on secondary factors like author or journal track records.
Graph-based methods like PageRank or disruption index use the context of the citing papers to evaluate an article's relevance [@doi:10.1073/pnas.0507655102; @jif; @pagerank; @doi:10.1038/s41586-019-0941-9].
Each of these methods has its strengths, and permutations exist that attempt to shore up specific weaknesses [@doi:10.1371/journal.pbio.1002541; @doi:10.1016/j.joi.2010.01.002; @doi:10.1007/s11192-017-2626-1; @doi:10.1162/qss_a_00068].

One objection to such practices is that "importance" is subjective.
The San Francisco Declaration on Research Assessment (DORA) argues against using Journal Impact Factor, or any journal-based metric, to assess individual manuscripts or scientists [@doi:10.1126/science.1240319].
DORA further argues in favor of evaluating the scientific content of articles and notes that any metrics used should be article-level (https://sfdora.org/read/).
However, even article-level metrics often ignore that the importance of a specific scientific output will fundamentally differ across fields.
Even Nobel prize-winning work may be unimportant to a cancer biologist if the prize-winning article is about astrophysics.

Because there are differences between fields' citation practices [@doi:10.1017/S0269889720000022], scientists have developed strategies including normalizing the number of citations based on nearby papers in a citation network, rescaling fields' citation data to give more consistent PageRank results, and so on [@doi:10.1371/journal.pbio.1002541; @doi:10.1007/s11192-020-03406-8; @doi:10.1016/j.joi.2017.05.014; @doi:10/f48tvt].
Such approaches normalize away field-specific effects, which might help to compare one researcher with another in a very different field.
However, they do not address the difference in the relevance of a topic between fields.
This phenomenon of field-specific importance has been observed at the level of journal metrics.
Mason and Singh recently noted that depending on the field, the journal *Christian Higher Education* is either ranked as a Q1 (top quartile) journal or a Q4 (bottom quartile) journal [@doi:10.1007/s11192-022-04402-w].

It is possible that, while global journal-level metrics fail to capture field-specific importance, article-level metrics are sufficiently granular that the importance of a manuscript remains constant across fields.
We investigate the extent to which article-level metrics generalize between fields.
We examine this using MeSH terms to define fields and use field-specific citation graphs to assess their importance within the field.
While it is trivially apparent that journals or articles that do not have cross-field citations will have variable importance, we ignore these cases and include only those with citations in both fields, where we expect possible consistency.
We first replicate previous findings that journal-level metrics can differ substantially among fields.
We also find field-specific variability in importance at the article level.
We make our results explorable through a web app that shows metrics for overlapping papers between pairs of fields.

Our results show that even article-level metrics can differ substantially among fields.
We recommend that metrics used for assessing research outputs include field-specific, in addition to global, ones.
While qualitative assessment of the content of manuscripts remains time-consuming, our results suggest that within-field and across-field assessment remains key to assessing the importance of research outputs.


## Results

### Journal rankings differ between fields

In an attempt to quantify the relative importance of journals, scientists have created rankings using metrics the Journal Impact Factor, which is essentially based on citations per article, and those that rely on more complex representations like Eigenfactor [@doi:10.5860/crln.68.5.7804]. 
It has previously been reported that journal rankings differ substantially between fields using metrics based on citation numbers [@doi:10.1007/s11192-022-04402-w].
We calculated a PageRank-based score for the journal as the median PageRank of manuscripts published in that journal for that field (Fig. @fig:journal A).
We first sought to understand the extent to which journal ranking differences replicated using PageRank.

To begin, we compared the differences in ranking between the top fifty journals in nanotechnology and their corresponding ranks in microscopy.
While the ranks were somewhat correlated there was a great deal of variance, especially for journals outside the top 20 in nanotechnology (Fig. @fig:journal B).
We then made use of the scale of the data by examining the top ranked journal in each of our 45 fields to determine whether the top ranking journal would be consistent across fields (Fig. @fig:journal C).
We found that the most common top-ranked journal was *Science*.
This was unsurprising, given that it tends to rank highly among global journal level metrics such as eigenfactor.
However, the ranking was very field-dependent, with only 20% of fields having *Science* as their top ranked journal.

One could argue that while general journals may have differing influence by field, specialty journals correspond to a single field so field-aware metrics are irrelevant.
That turns out to be untrue.
Of the 5,178 journals with at least 50 articles present in our dataset, the median number of fields publishing in a given journal is 15 (Fig. @fig:journal D).
This result confirms that while useful [@carlsson2009], MeSH headings reflect a different type of aggregation than journals do [@doi:10.1007/s11192-016-2119-7].

![ 
Journals' PageRank derived rankings differ between fields. 
A) A schematic showing how paired networks are derived from the full citation network.
B) A comparison of the ranks of the top 50 journals by PageRank in nanotechnology and their rank in microscopy. 
Top-50 nanotechnology journals with no papers in microscopy have been omitted.
C) The frequency with which journals in the dataset are the top journal for a field.
D) The distribution of fields published per journal. 
The X-axis corresponds to the number of fields for which a journal has at least one paper within the field.
All plots restrict the set of journals to those with at least 50 papers in the dataset.

](./images/journal_fig.png ){#fig:journal width="100%"}

### Manuscript PageRanks differ between fields
We split the citation network into its component fields and calculated the PageRank for each article (Fig. @fig:distribution A).
We then examined the distribution of PageRanks across fields and found that they differed greatly (Fig. @fig:distribution B).
These differences were driven by the size and citation practices of the fields themselves, as the papers shared by pairs of fields had distributions matching the field context they were in (Fig. @fig:distribution B, C).
Given the differences in distributions in articles shared by these fields (Fig. @fig:distribution D), we found it difficult to determine whether correspondance between fields was random or due to different degrees of interest in certain articles (Fig. @fig:distribution E).

![ 
Differences in the distribution of PageRanks between fields.
A) A schematic showing how field pairs are split and their PageRanks are calculated.
B) The distribution of article PageRanks for nanotechnology and microscopy. 
The distributions marked with 'All' contain all the papers for the given field in the dataset, while those marked 'overlapping' contain only articles present in both fields.
C) The empirical cumulative density functions of nanotechnology and microscopy.
D) The differences in distribution of the PageRanks of articles shared by nanotechnology and microscopy.
E) A density plot showing the joint distribution of PageRanks for papers overlapping in nanotechnology and microscopy.
](./images/distribution_fig.png ){#fig:distribution width="100%"}

### Fields' differences are not solely driven by differences in citation practices
We devised a strategy to generate an empirical null for a field pair under the assumption that the field pair represented a single, homogenous field (Fig. @fig:percentile A).
For each field-pair intersection, we performed a degree-distribution preserving permutation.
We created 100 permuted networks for each field pair.
We then split the networks into their constituent fields and calculated a percentile using the number of permuted networks with a lower PageRank for a manuscript than the true PageRank.
A manuscript with a PageRank higher than all networks has a percentile of 100, and one lower than all permuted networks has a percentile of zero. 
We used the difference in the percentile in each field as the field-specific affinity for a given paper.
This percentile score allowed us to control for the differing degree distributions between fields by comparing papers based on their expected PageRank in a random network with the same node degrees.

We selected field pairs with varying degrees of correlation between their PageRanks (Fig. @fig:percentile B).
By examining the fields' PageRank percentiles, we found that many articles had large differences in their perception between fields (Fig. @fig:percentile C).
In nanotechnology and microscopy, papers with high nanotechnology percentiles and low microscopy percentiles tended towards applications of nanotechnology, while their counterparts with high microscopy percentiles and low nanotechnology percentiles were often papers about technological developments in microscopy (Fig. @fig:percentile A, Table 1).
Immunochemistry-favored papers are largely applications of immunochemical methods, while anatomy-favored articles tend to focus experiments on a single anatomical region (Fig. @fig:percentile B, Table 2).
Proteomics and metabolomics tend to use similar methods, so the fields on either end are largely (though not entirely) field-specific applications of those methods (Fig. @fig:percentile C, Table 3).
Computational biology is similarly applications-focused, though human genetics tends towards policy papers due to its MeSH heading (H01.158.273.343.385) excluding fields like genomics, population genetics, and microbial genetics (Fig. @fig:percentile D, Table 4).
In addition to papers with large differences between fields, each field also has papers with high PageRanks and similar percentiles in both fields.
Overall it is clear that while some papers may be influential in multiple fields, others have more field-specific import.

It is not possible to describe all the field-pairs and relevant differences between fields within the space of a journal article.
Instead, we have developed a web server that displays the percentiles for all pairs of fields in our dataset with at least 1000 shared articles (Fig. @fig:percentile D), which can be accessed at https://www.indices.greenelab.com.
We hope that the availability of the web server and the reproducibility of our code will assist other scientists in uncovering new insights from this dataset.

![ 
Field-specific preferences in papers.
A) A schematic showing how networks are shuffled and how articles' percentile scores are calculated.
The histograms at the bottom of the figure correspond to the distribution of PageRanks for the shuffled networks, while the red lines correspond to an article's PageRank in the true citation network.
B) The Pearson correlation of PageRanks between fields. 
The red points are the field pairs expanded in panel C.
C) The percentile scores and PageRanks for overlapping articles in various fields. 
Points are colored based on the difference in percentile scores in the fields e.g. "Nanotechnology-Microscopy" corresponds to the difference between the nanotechnology and microscopy percentile scores.
The numbers next to points are the reference number for the article in the bibliography.
D) A screenshot of the webserver showing the percentile score difference and journal median PageRank plot functionality.
](./images/percentile_figure.png ){#fig:percentile width="100%"}

| Nanotechnology Percentile | Microscopy Percentile | Title | Reference |
|----------|----------|----------|----------|
| 100 | 4  | A robust DNA mechanical device controlled by hybridization topology | [@doi:10.1038/415062a] |
| 100 | 5  | Bioadhesive poly(methyl methacrylate) microdevices for controlled drug delivery | [@doi:10/c7fpg4] |
| 99  | 2  | DNA-templated self-assembly of protein arrays and highly conductive nanowrires | [@doi:10.1126/science.1089389] |
| 0   | 100| Photostable luminescent nanoparticles as biological label for cell recognition of system lupus erythematosus patients | [@doi:10.1166/jnn.2002.105] |
| 5   | 90 | WSXM: a software for scanning probe microscopy and a tool for nanotechnology | [@horcas2007wsxm] |
| 0   | 77 | Measuring Distances in Supported Bilayers by Fluorescence Interference-Contrast Microscopy: Polymer Supports and SNARE Proteins | [@doi:10/dqsg2c] |
| 100 | 99 | Toward fluorescence nanoscopy | [@doi:10.1038/nbt895] |
| 100 | 86 | In vivo imaging of quantum dots encapsulated in phospholipid micelles | [@doi:10.1126/science.1077194]|
| 100 | 99 | Water-Soluble Quantum Dots for Multiphoton Fluorescence Imaging in Vivo | [@doi:10.1126/science.1083780]|

**Table 1**: Nanotechnology/microscopy papers of interest 

| Immunochemistry Percentile| Anatomy Percentile | Title | Reference |
|----------|----------|----------|----------|
| 100 | 45 | Immunoelectron microscopic exploration of the Golgi complex | [@immunoelectron] |
| 100 | 14 | Immunocytochemical and electrophoretic analyses of changes in myosin gene expression in cat posterior temporalis muscle during postnatal development | [@doi:10.1007/bf01682147] |
| 98 | 5 | Electron microscopic demonstration of calcitonin in human medullary carcinoma of thyroid by the immuno gold staining method | [@doi:10.1007/bf00514331] |
| 12 | 100 | Grafting genetically modified cells into the rat brain: characteristics of E. coli β-galactosidase as a reporter gene | [@doi:10/dptnbm] |
| 12 | 100 | Vitamin-D-dependent calcium-binding-protein and parvalbumin occur in bones and teeth | [@doi:10.1007/bf02405306] |
| 3 | 100 | Mapping of brain areas containing RNA homologous to cDNAs encoding the alpha and beta subunits of the rat GABAA gamma-aminobutyrate receptor | [@doi:10.1073/pnas.85.20.7815] |
| 100 | 100 | Studies of the HER-2/neu Proto-Oncogene in Human Breast and Ovarian Cancer| [@her2] |
| 100 | 100 | Expression of c-fos Protein in Brain: Metabolic Mapping at the Cellular Level| [@doi:10.1126/science.3131879] |
| 100 | 100 | Proliferating cell nuclear antigen (PCNA) immunolocalization in paraffin sections: An index of cell proliferation with evidence of deregulated expression in some neoplasms| [@doi:10.1126/science.3131879] |

**Table 2**: Immunochemistry/anatomy papers of interest 

| Proteomics Percentile | Metabolomics Percentile | Title | Reference |
|----------|----------|----------|----------|
| 67 | 2 | Proteomics Standards Initiative: Fifteen Years of Progress and Future Work | [@doi:10.1021/acs.jproteome.7b00370] |
| 99 | 0 | Limited Environmental Serine and Glycine Confer Brain Metastasis Sensitivity to PHGDH Inhibition | [@doi:10.1158/2159-8290.cd-19-1228] |
| 100 | 0 | A high-throughput processing service for retention time alignment of complex proteomics and metabolomics LC-MS data| [@doi:10.1093/bioinformatics/btr094 ] |
| 0 | 100 | MeltDB: a software platform for the analysis and integration of metabolomics experiment data| [@doi:10.1093/bioinformatics/btn452] |
| 0 | 98 | In silico fragmentation for computer assisted identification of metabolite mass spectra| [@doi:10.1186/1471-2105-11-148] |
| 0 | 100 | The Metabonomic Signature of Celiac Disease| [@doi:10.1021/pr800548z] |
| 91 | 70 | Visualization of omics data for systems biology| [@doi:10.1186/1471-2105-11-148] |
| 0 | 16 | FunRich: An open access standalone functional enrichment and interaction network analysis tool| [@doi:10.1002/pmic.201400515] |
| 0 | 5 | Proteomic and Metabolomic Characterization of COVID-19 Patient Sera| [@doi:10.1016/j.cell.2020.05.032] |

**Table 3**: Proteomics/metabolomics papers of interest 

| Computational Biology Percentile | Human Genetics Percentile | Title | Reference |
|----------|----------|----------|----------|
| 99 | 0 | Development of Human Protein Reference Database as an Initial Platform for Approaching Systems Biology in Humans| [@doi:10.1101/gr.1680803] |
| 100 | 1 | A database for post-genome analysis| [@doi:10/cfgb98] |
| 100 | 1 | Use of mass spectrometry-derived data to annotate nucleotide and protein sequence databases| [@doi:10/ch565r] |
| 12 | 100| Genetic Discrimination: Perspectives of Consumers | [@doi:10.1126/science.274.5287.621] |
| 0 | 81 | Committee Opinion No. 690: Carrier Screening in the Age of Genomic Medicine | [@committee] |
| 23 | 100 | Public health genomics: The end of the beginning| [@doi:10.1097/gim.0b013e31821024ca] |
| 100 | 99 | Initial sequencing and analysis of the human genome| [@doi:10.1038/35057062] |
| 100 | 100 | An STS-Based Map of the Human Genome| [@doi:10.1126/science.270.5244.1945] |
| 100 | 100 | A New Five-Year Plan for the U.S. Human Genome Project| [@doi:10.1126/science.8211127] |

**Table 4**: Computational biology/human genetics papers of interest 



## Methods

#### COCI
We used the March 2022 version of the COCI citation index [@doi:10.1007/s11192-019-03217-6] as the source of our citation data.
This dataset contains around 1.3 billion citations from ~73 million bibliographic resources.

#### Selecting fields
To differentiate between scientific fields, we needed a way to map papers to fields.
Fortunately, all the papers in Pubmed Central (https://www.ncbi.nlm.nih.gov/pmc/) have corresponding Medical Subject Headings (MeSH) terms.
While MeSH terms are varied and numerous, the subheadings of the Natural Science Disciplines (H01) category fit our needs.
However, MeSH terms are hierarchical, and vary greatly in their size and specificity.
To extract a balanced set of terms we recursively traversed the tree and selected headings that have least 10000 DOIs and don't have multiple children that also meet the cutoff.
Our resulting set of headings contained 45 terms, from "Acoustics" to "Water Microbiology".

#### Handling citation networks
The COCI dataset consists of pairs of Digital Object Identifiers (DOIs).
To change these pairs into a form we could run calculations on, we needed to convert them into networks.
To do so, we created 45 empty networks, one for each MeSH term we selected previously.
We then iterated over each pair of DOIs in COCI, and added them to a network if the DOIs corresponded to two journal articles written in english, both of which were tagged with the corresponding MeSH heading.

Because we were interested in the differences between fields, we also needed to build networks from pairs of MeSH headings.
These networks were built via the same process, except that instead of keeping articles corresponding to a single DOI we added a citation to the network if both articles were in the pair of fields, even if the citation occurred across fields.
Running this network-building process yielded 990 two-heading networks.

Sampling a graph from the degree distribution while preserving the distribution of degrees in the network turned out to be challenging.
Because citation graphs are directed, it's not possible to simply swap pairs of edges and end up with a graph that is uniformly sampled from the space.
Instead, a more sophisticated three-edge swap method must be used [@arxiv:0905.4913].
Because this algorithm had not been implemented yet in NetworkX [@networkx], we wrote the code to perform shuffles and submitted our change to the library.
With the shuffling code implemented, we created 100 shuffled versions of each of our combined networks to act as a background distribution to compare metrics against.

Once we had a collection of shuffled networks, we needed to split them into their constituent fields.
To do so, we reduced the network to solely the nodes that were present in the single heading citation network, and kept only citations between these nodes.

#### Metrics

We used the NetworkX implementation of PageRank with default parameters to evaluate paper importance within fields.
To determine the degree to which the papers' PageRank values were higher or lower than expected, we compared the PageRank values calculated for the true citation networks to the values in the shuffled networks for each paper.
We then recorded the fraction of shuffled networks where the paper had a lower PageRank than in the true network to derive a single number that described these values.
For example, if a paper had a higher PageRank in the true network than in all the shuffled networks it received a score of 1.
Likewise, if it had a lower PageRank in the true network than in all the shuffled networks it received a score of 0.
Papers in between the two extremes had fractional values, like .5 (a paper that fell in the middle of the pack) and so on.

A convenient feature of the percentile scores is that they're directly comparable between fields.
If a paper is present in two fields, the difference in scores between the two fields can be used to estimate its relative importance.
For example, if a paper has a score of 1 in field A (indicating a higher PageRank in the field than expected given its number of citations and the network structure) and a score of 0 in field B (indicating a lower than expected PageRank), then the large difference in scores indicates the paper is more highly valued in field A than field B.
If the paper has similar scores in both fields, it indicates that the paper is similarly valued in the two fields.

#### Hardware/runtime
The analysis pipeline was run on the RMACC Summit cluster.
The full pipeline, from downloading the data to analyzing it to vizualizing it took about a week to run.
However, that number is heavily dependent on details such as the number of CPU nodes available and the network speed.

Our webserver is built by visualizing our data in Plotly (https://plotly.com/python/plotly-express/) on the Streamlit platform (https://streamlit.io/).
The field pairs made available by the frontend are those with at least 1000 shared papers after filtering out papers with more than a 5% missingness level of their PageRanks after shuffling.
The journals available for visualization are those with at least 25 papers for the given field pair.


## Discussion/Conclusion

We analyze hundreds of field-pair citation networks to examine the extent to which article-level importance metrics vary between fields.
As previously reported, we find systematic differences in PageRanks between fields [@doi:10.1007/s11192-017-2626-1; @doi:10.1007/s11192-014-1308-5] that would warrant some form of normalization when making cross-field comparisons with global statistics.
However, we also find that field-specific differences are not driven solely by differences in citation practices.
Instead, the importance of individual papers appears to differ meaningfully between fields.
Global rankings or efforts to normalize out field-specific effects obscure meaningful differences in manuscript importance between communities.

As with any study, this research has certain limitations.
One example is our selection of MeSH terms to represent fields.
We used MeSH because it is a widely-annotated set of subjects in biomedicine and thresholded MeSH term sizes to balance having enough observations to calculate appropriate statistics with having sufficient granularity to capture fields.
This selection process resulted in fields at the granularity of "biophysics" and "ecology."
We also have to select a number of swaps to generate a background distribution of PageRanks for each field pair.
We selected three times as many swaps as edges, where each swap modifies three edges, but certain network structures may require a different number.

We also note that there are inherent issues with the premise of ranking manuscripts' importance.
We sought to understand the extent to which such rankings were stable between fields after correcting for field-specific citation practices.
We found limited stability between fields, mostly between closely-related fields, suggesting that the concept of a universal ranking of importance is difficult to justify.
In the way that reducing a distribution to a Journal Impact Factor distorts assessment, attempting to use a single universal score to represent importance across fields poses similar challenges at the level of individual manuscripts.
Furthermore, this work's natural progression would extend to estimating the importance of individual manuscripts to individual researchers.
Thus, a holistic measure of importance would need to include a distribution of scores not only across fields but across researchers.
It may ultimately be impossible to calculate a meaningful importance score.
The lack of ground truth for importance is an inherent feature, not a bug, of science's step-wide progression.

Shifting from the perspective of evaluation to discovery can reveal more appropriate uses for these types of statistics.
Field-pair calculations for such metrics may help with self-directed learning of new fields.
An expert in one field, e.g., computational biology, who aims to learn more about genetics may find manuscripts with high importance in genetics and low importance in computational biology to be important reads.
These represent manuscripts not currently widely cited in one's field but highly influential in a target field.
Our application can reveal these manuscripts for MeSH field pairs, and our source code allows others to perform our analysis with different granularity.


### Code and Data Availability                                                                                                                                                                              
The code to reproduce this work can be found at https://github.com/greenelab/indices.
The data used for this project is publicly available and can be downloaded with the code provided above. 
Our work meets the bronze standard of reproducibility [@doi:10.1038/s41592-021-01256-7] and fulfills aspects of the silver and gold standards including deterministic operation.

### Acknowledgements                                                                                                                                                                                        
We would like to thank Jake Crawford for reviewing code that went into this project and Faisal Alquaddoomi for figuring out the web server hosting.
We would also like to thank the past and present members of GreeneLab who gave feedback on this project during lab meetings.
This work utilized resources from the University of Colorado Boulder Research Computing Group, which is supported by the National Science Foundation (awards ACI-1532235 and ACI-1532236).

#### Funding
This work was supported by grants from the National Institutes of Health’s National Human Genome Research Institute (NHGRI) under award R01 HG010067 and the Gordon and Betty Moore Foundation (GBMF 4552) to CSG.
The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript. 


# Chapter 6 - Future Directions

**Contributions:**
I wrote and edited this chapter for the purpose of being included in this dissertation.

## Introduction

In this dissertation, we have examined whether deep learning has led to a paradigm shift in computational biology.
We established standards for reproducible research when using deep learning models in chapter 2, showed that deep learning is not always preferable to other techniques in chapter 3, then demonstrated the effectiveness of classical ml methods in chapters 4 and 5.
Ultimately we concluded that while deep learning has been a useful tool in some areas, it has yet to lead to a paradigm shift in computational biology.
However, deep learning models' impact may grow as the fields develop, so we would like to discuss future areas where we expect interesting developments.

## Deep learning representations of biology

Different areas of computational biology research have seen different effects from deep learning.
Deep learning has already had a significant impact on biomedical imaging [@doi:10.1016/j.csbj.2020.08.003], and seems poised to do so in protein structure [@doi:10.1038/s41586-021-03819-2].
These advances were likely successful because of their similarity to well-researched fields in that they can be framed as similar problems.
Biomedical images are not the same as those from a standard camera, but the inductive bias of translational equivariance and various image augmentation methods are still applicable.
Similarly, while protein sequences may not seem to share much with written language, models like RNNs and transformers that look at their input as a sequence of tokens do not care whether those tokens are words or amino acids.

Not all subfields of computational biology have convenient ways to represent their data, though.
Gene expression, in particular, is difficult because of its high dimensionality.
Expression data does not have spatial locality to take advantage of, so convolutional networks cannot be used to ignore it.
It is not a series of tokens either; the genes in an expression dataset are listed lexicographically, so their order does not have meaning.
Self-attention seems well-suited for gene expression since learning which subsets of genes interact with others would be useful. 
The high dimensionality makes vanilla self-attention infeasible though, due to the quadratic scaling.
This issue cannot even be sidestepped with standard dimensionality reduction methods without losing predictive performance.

Do any deep learning representations work for gene expression, then?
Fully-connected networks work, though they do not tend to be the best way to accomplish most tasks.
An interesting potential research direction would be to apply sparse self-attention methods to gene expression data and reduce the number of comparisons made by only attending within prior knowledge gene sets.
Alternatively, because expression is often thought of in terms of coregulation networks or sets of genes with shared functions, a graph representation may be more suitable.
It is also possible that someone will develop a representation specifically for gene expression that will work better than anything we know about today.

## To what extent is biology limited by challenges in looking at the data

An essential first step when working with data is to look at it.
In images or generated text, a human can judge how good generated data is.
In the classification world, a human labeler can look at an image and say, "that is a dog," or a sentence and say, "that is grammatically correct English."
While these labels are somewhat fuzzy, a group of humans can at least look at the label and say, "that is reasonable" or "that is mislabeled."
A human looking at a gene expression microarray or a table of RNA-seq counts is cannot do the same.

Our brains are built to recognize objects, not parse gene expression perturbations corresponding to septic shock.
This issue is not insurmountable; scientists can do research in quantum physics, after all.
It simply serves as a hindrance to our ability to sanity-check data.
Because we cannot see whether the relevant signals are distorted by batch effect normalization or a preprocessing step, we must be more careful and try more options.
Perhaps in the future, as we understand more about the relevant biology, scientists will be able to create views of the data that are more human-intuitive and easier to use.

## The scale of biological data

Biological data (or at least transcriptomic data) is not actually that big.
The largest uniformly processed compendia of bulk human expression data contain hundreds of thousands of samples.
Meanwhile in machine learning, even before deep learning took off ImageNet already had more than three million images [@doi:10.1109/CVPR.2009.5206848].

Worse, many biological domains have strict upper bounds on the amount of data available.
Even if one somehow recruited the entire world for a study, they would only be able to collect around eight billion human genomes.
Given the complexity of biology, it seems unlikely that "only" eight billion genomes would be sufficient to effectively sample the space of plausible relevant mutations in the human genome.
Based on recent research into neural network scaling laws [@arxiv:2203.15556] and machine learning intuition, it seems likely that Rich Sutton's "Bitter Lesson" (http://www.incompleteideas.net/IncIdeas/BitterLesson.html) would break down in a domain where there is a hard cap on the available data.
This data cap probably is not true of all domains in computational biology, though.
Gene expression changes with variables like cell type, time, and biological state, so the space of transcriptomic data that could be measured is much larger.

While we have shown that deep learning has not led to a paradigm shift in computational biology so far, will that always be true? 
As with many scientific questions, the answer is probably "it depends."
While there may be caps on individual aspects of biological data, there are always more angles of attack.

The promise of multiomics has always been that multiple views of the same system may reveal something that no single view picks up.
The challenge is that the data types are different, their relationships are not well-characterized, and the methods for working in such a system have not been fully developed yet.
Transformer architectures, and more specifically their self-attention mechanism, seem like a good fit for learning relationships between different 'omes.
Such models are data-hungry, though, and self-attention gets expensive in problems with high dimensionality.
Perhaps one day we will have the data and compute to train multiomic biological transformers.
Or maybe by then the state of the art in machine learning will have moved along, rendering them irrelevant.

## Conclusion
Whether deep learning takes over or simply becomes another tool in our toolbelt, the future of computational biology looks bright.
These are exciting times indeed.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
