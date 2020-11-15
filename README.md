# Does Causal Coherence Predict Online Spread of Social Media?

[![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1007/978-3-030-21741-9_19)

This repository is dedicated to all of the documents, codes, data, results, and reproducibility report for our paper titled "Does Causal Coherence Predict Online Spread of Social Media?" which is accepted for publication at the SBP-BRiMS 2019 conference.

## Input dataset
The input files to our analysis are news documents from [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) dataset. It is worth pointing out that FakeNewsNet has been recently updated and new articles are now added to the dataset. However, in this paper we used the old version of the dataset --since we had already done our analysis at the time of paper submission-- which can be found on FakeNewsNet repository under the **old-version** branch.

## Pre-processing

We processed all the news documents in the dataset and stored them a reformatted version in an excel file which can found at ```data/FakeNewsNet/old/processed/fakenewsnet.csv```. A list of news documents' ids that we excluded from our analysis can be found in ```trash.txt``` at ```data/FakeNewsNet```. We excluded these documents since they did not have a proper value in their body text of the article thus useless for Coh-metrix analysis. 


## Applying Coh-metrix
We created one single text file for each news document as the input for [Coh-metrix](http://cohmetrix.com/). We used Coh-metrix for computing computational cohesion and coherence metrics for the news documents. The output of Coh-metrix is a single csv file named ```fakenewsnet_coh_un.csv``` which can be found in this repository at ```data/FakeNewsNet/old/cohmetrix/cohout```

At the end, we created a final excel file named ```fakenewsnet_full_un.xlsx``` which is the input to our regression analysis and can be found at ```data/FakeNewsNet/old/processed```. This file includes the Coh-metrix indexes for all the news documents in addition to the following columns:

* ```label```: truth label of the news article from FakeNewsNet which can be either fake or real. "1" for fake, and "0" for real.
* ```shares```: the number of distinct users who shared a news article. It is important to mention that in FakeNewsNet, each user can share the same news article multiple times; however, in our analysis, we considered the distinct number of users who shared a news article as the count of shares of the news article.
* ```total_shares```: the total number of shares of a news document including users who shared the news document multiple times.
* ```checker```: the source of fact checking the news documents. "1" for BuzzFeed and "2" for PolitiFact.
* ```id```: id of the news article which matches the id columns in the ```fakenewsnet.csv```

## Regression analysis
All the regression analysis that we did in R can be found in a jupyter notebook named ```experiments.ipynb```.

### Citation
Please use the following information to cite our paper:

```bibtex
@inproceedings{hosseini2019does,
  title={Does Causal Coherence Predict Online Spread of Social Media?},
  author={Hosseini, Pedram and Diab, Mona and Broniatowski, David A},
  booktitle={International Conference on Social Computing, Behavioral-Cultural Modeling and Prediction and Behavior Representation in Modeling and Simulation},
  pages={184--193},
  year={2019},
  organization={Springer}
}
