## Keyword-Spotting with Dynamic Time-Warping

#### Methods
We started by preprocessing our data. We converted all pages to a binarized image using Otsu's Method. 
We continued preparation by extracting each word from each page by iterating through all polygons, which could be found in the `.svg`-file, 
and cutting the images accordingly. Preprocessing steps can be seen in the documents `placeholder otsu` and `placeholder svg`. 
The single words will then have following format: 

IMAGE OF SINGLE WORD

The next step was to extract features for each word. We decided to use seven different features for each word, which are as follows:
* Number of black/white transitions
* Fraction of black pixels
* Position of upper contour
* Position of lower contour
* Fraction of black pixels within contours
* Gradient of upper contour
* Gradient of lower contour

As can be seen in the image of the single word, regions around the polygon are black, and thus have to be removed from the image 
before applying. Else, they would be registered as being part of the word.
We iterate through each word using a rolling window of width = 1px and calculate 7 features for each window.
These 7 resulting lists are saved as a `.txt`-file. Feature extraction can be seen in `placeholder features`.

For applying the Dynamic Time-warping algorithm (DTW), we chose a specific keyword and selected images of the
training set containing the keyword. We limited the amount of training words to five due to computational cost.
We ignored punctuation, which means we regard 'letters' and 'letters,' as the same word.
Afterwards, each image of the validation set is compared to those selected images and the algorithm is applied between them. 
We applied the euclidean distance to measure the distance between two features, and used a Sakoe-Chiba band of size = 10 to reduce
the computational cost. The created matrix of distances can then be used to calculate the over-all distance, by simply taking the last
value of the matrix. We thus get seven distances representing each feature for each image-pair.

We then sort the distances and thus get a ranking. We take each rank and apply a majority voting, taking the image that is found the most often at each rank. We then draw a P/R-curve, calculating true-negative, true-positive and false-positive counts for
increasing amounts of evaluated ranks. 

After getting some results, we decided to implement the library `fastdtw` to increase the computational speed from aprroximately
5 hours to 30 minutes per word. We also adjusted our ranking. We now rank one feature, calculate all counts needed for a P/R-curve 
and take the mean between all features of a word.
DTW and ranking can be seen in `placeholder DTW`, plotting can be seen in `placeholder plotting`.

#### Results
Application of the algorithms for selected words can be seen below. We show three P/R-curves done with our own DTW algorithm and the
old ranking system, and two curves done with `fastdtw` and the new ranking system:
* 'with' (old)
* 'and' (old)
* 'Williamsburghs' (old)
* '' (new)
* '' (new)

PLACEHOLDER IMAGES:

#### Discussion
During testing, we encountered issues with performance. We thus decided to compare each validation image with some selected images from
the training set, rather than comparing each validation image to the whole training set, which decreases computation time drastically.
It also leads to less true-positives, causing the curve to be less smooth. Counting each appearance of a word in the ranked feature lists
and weight them by their rank could also improve evaluation, but this would also be computationally heavier.

Some applications return P/R-curves of a shape that we more or less expected (see plots for 'with' and 'and'), and some curves show no 
significant keyword detection (see plot for 'Williamsburgh'). This correlates with the amount of appearances of the chosen keyword in 
the validation set. The more words in the validation set, the more likely it is to get a high score for some of them. Also, if there
is only one reference word in the training set, the scoring will be less accurate.

Switching to the `fastdtw`-library and adjusting the ranking resulted in much smoother results. The accuracy is still low, but the performance does not vary as much. Adding more training words also improves the accuracy.




