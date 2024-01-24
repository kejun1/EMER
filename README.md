# EMER
This is a Pytorch implementation of "Smile upon the Face but Sadness in the Eyes: Emotion Recognition based on Facial Expressions and Eye Behaviors"
## Details
Th Eye-behavior-aided Multimodal Emotion Recognition (EMER) employs a stimulus material-induced spontaneous emotion generation method to integrate non-invasive eye behavior data, like eye movements and eye fixation maps, with facial videos collected from 1,303 samples involving 121 participants, aiming to obtain natural and accurate human emotions. Most importantly, the EMER introduces comprehensive annotation information. These annotations include  3-class coarse ER and FER labels (namely positive, negative and neutral), 7-class fine ER and FER labels (namely happiness, sadness, fear, surprise, disgust, anger, and neutral), 2-dimensional continuous emotion ratings (namely valence and arousal), as well as facial expression intensity. This extensive range of data and annotations makes the EMER highly suitable for real-world applications. EMER database has enormous diversities, large quantities, and rich annotations, including:

* 1303 number of data from 121 participants.
* Comprehensive annotation information:
  * 3-class coarse ER and FER labels, i.e., positive, negative and neutra.
  * 7-class fine ER and FER labels , i.e., happiness, sadness, fear, surprise, disgust, anger, and neutral.
  * Facial expression intensit for each discrete category in the range [0,3].
  * Two-dimensional continuous emotion ratings, i.e., valence and arousal ratings in the range [-1,1].
* three subsets: facial expression subset, eye movement subset, and the eye fixation map subset.

## Example
<div align="center">
<img src="./data/example.png" width="700px">
</div>

## Terms & Conditions
1. EMER database is available for non-commercial research purposes only.
2. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for commercial purposes, any portion of the clips, and any derived data.
3. You agree not to further copy, publish, or distribute any portion of the EMER database. Except for internal use at a single site within the same organization, it is allowed to make copies of the dataset.

## How to get the EMER Dataset
This database is publicly available. It is free for professors and researcher scientists affiliated to a University.

Permission to use but not reproduce or distribute the EMER database is granted to all researchers given that the following steps are properly followed:

1. Download the `./data/EMER academics-final.pdf` document.
2. Read the terms and conditions carefully to make sure they are acceptable, and fill in the relevant information at the end of the document.
3. Send the completed document to email.

## Content

* `./checkpoints`: Contains trained models.
* `./data`: Data storage location, contains database application pdf.
* `./dataset`: Contains database read files.（After requesting EMER database we will provide the database read file.）
* `discrete_test_EMERT.py`: Emotion classification evaluation code.
* `valence_test_EMERT.py`: Valence regression evaluation code.
* `arousal_test_EMERT.py`: Arousal regression evaluation code.
* `intensity_test_EMERT.py`: Emotional intensity regression evaluation code.

## Requirements
* python 3.7.9
* Pytorch 1.7.1
* opencv-python 4.5.1

## Model
We provide trained model in `./checkpoints`.

## Usage
First, prepare EMER data and put it in the folder `./data`. 
### Testing
We provide trained models in `./checkpoints` and evaluation codes for three tasks.
#### For Multimodal Emotion Classification
```
cd EMER
python discrete_test_EMERT.py
```
#### For Multimodal Valence and Arousal Regression
```
cd EMER
python valence_test_EMERT.py
python arousal_test_EMERT.py
```
#### For Multimodal Emotional Intensity Regression 
```
cd EMER
python intensity_test_EMERT.py
```
