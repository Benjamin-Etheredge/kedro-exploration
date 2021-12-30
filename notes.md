# TODO
- Do i have to nuke sql dbs every time I make a code change? Probably.
- ~~make cross val splits even per label~~ already does it internally
- figure out how to consistently get kedro install to make nice requirements.txt
- figure out testing and docs
- plot confusion matrixes
- explore visualization of decision tree and random forest
- explore what's being pruned in decision tree
- add analysis to node to each dataset
- add analysis to node to each report
- be sure to seed all classifiers
- had to tweak knn a lot to get it effeicent 
https://scikit-learn.org/stable/modules/neighbors.html#classification

## Important Notes
- shouldn't need probabilitys from svc. Something wrong in the metric
- add baseline versions for comparisons (no sweeping)
- there is a leak in the training data for cross val but not to test set

## SVM notes
From sklearn: "The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples. For large datasets consider using LinearSVC or SGDClassifier instead, possibly after a Nystroem transformer."