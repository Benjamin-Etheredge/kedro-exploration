from typing import Dict
from sklearn.metrics import classification_report
from sklearn.base import ClassifierMixin
from icecream import ic

# TODO break apart?
def grade(
    model: ClassifierMixin, 
    train_x, 
    train_y, 
    test_x, 
    test_y, 
) -> Dict[str, Dict[str, float]]:
    # Calculate the accuracy of the model
    train_y_hat = model.predict(train_x)
    test_y_hat = model.predict(test_x)

    ic(model.get_params())
    #acc = accuracy_score(train_y, train_y_hat)
    #print(acc)
    return dict(
        train=classification_report(train_y, train_y_hat, output_dict=True),
        test=classification_report(test_y, test_y_hat, output_dict=True),
        params=model.get_params(),
    )
 