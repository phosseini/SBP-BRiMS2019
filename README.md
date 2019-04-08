# SBP-BRiMS2019-paper

This reposiroty is dedicated to keeping all the codes, data, results from my paper submitted to the SBP-BRiMS 2019 conference.

## Input dataset
The input files to our analysis are the news documents from [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) dataset. It is worth poiting out that FakeNewsNet has been recently updated and new articles are now added to the dataset. However, in this paper we used the old version of this dataset --since we had already done our analysis at the time of paper submission-- which can be found on their repository under the **old-version** branch.

## Pre-processing

We processed all the news documents in the dataset and stored a reformatted version in an excel file which can found at ```data/processed/fakenewsnet.csv```. The method we defined for cleaning documents is called ```text_clean()``` which can be found in the ```pre_processing.py```. We then created one single text file for each news document as the input for [Coh-metrix](http://cohmetrix.com/). We used Coh-metrix for computing computational cohesion and coherence metrics for the news documents.



