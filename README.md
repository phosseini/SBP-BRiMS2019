# Does Causal Coherence Predict Online Spread of Social Media?

This reposiroty is dedicated to all the documents, codes, data, results, and reproducbility report for our paper which is accepted for publication at the SBP-BRiMS 2019 conference.

## Input dataset
The input files to our analysis are the news documents from [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) dataset. It is worth poiting out that FakeNewsNet has been recently updated and new articles are now added to the dataset. However, in this paper we used the old version of this dataset --since we had already done our analysis at the time of paper submission-- which can be found on their repository under the **old-version** branch.

## Pre-processing

We processed all the news documents in the dataset and stored a reformatted version in an excel file which can found at ```data/FakeNewsNet/processed/fakenewsnet.csv```. The method we defined for cleaning documents is called ```text_clean()``` which can be found in the ```pre_processing.py```. A list of news documents' ids that we excluded from our analysis can be found in ```trash.txt``` at ```data/FakeNewsNet/trash.txt```. We excluded these documents since they did not have a proper value in their body text of the article thus uselsess for Coh-metrix. 


## Applying Coh-metrix
We then created one single text file for each news document as the input for [Coh-metrix](http://cohmetrix.com/). We used Coh-metrix for computing computational cohesion and coherence metrics for the news documents. The output of Coh-metrix is a single csv file named ```fakenewsnet_coh.csv``` which can be found in this repository at ```data/FakeNewsNet/cohmetrix/cohout/fakenewsnet_coh.csv```

At the end, we created a final excel file named ```fakenewsnet_full.xlsx``` which is the input to our regression analysis and can be found at ```data/FakeNewsNet/processed/fakenewsnet_full.xlsx```. This file includes the Coh-metrx indexes for all the news documents in addition to the three following columns:

* ```label```: truth label of the news article from FakeNewsNet which be either fake or real.
* ```shares```: the number of disticst users who shared a news article. It is important ot mention that in FakeNewsNet, each user can share the same news article multiple times; however, in our analysis, we considered the distinct number of users who shared a news article as the count of shares of the news article. 
* ```id```: id of the news article which matched the id columns in the ```fakenewsnet.csv```

## Regression analysis
All the regression analysis that we did in R can be found in a jupyter notebook named ```regression_analysis.ipynb```.

### Citation
Please use the following information to cite our paper:

```
@inproceedings{hosseini2019causalcoherence,
               title={Does Causal Coherence Predict Online Spread of Social Media?},
               author={Hosseini, Pedram and Diab, Mona and Broniatowski, David A.},
               booktitle={International Conference on Social Computing, Behavioral-Cultural Modeling and Prediction and Behavior Representation in Modeling and Simulation (SBP-BRiMS).},
               year={2019},
               organization={Springer.}
}
