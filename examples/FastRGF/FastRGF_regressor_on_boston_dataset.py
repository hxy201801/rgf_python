from sklearn.datasets import load_boston
from sklearn.utils.validation import check_random_state
from rgf.sklearn import FastRGFRegressor

boston = load_boston()
rng = check_random_state(42)
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

for i in range(1, 100):
    train_x = boston.data[:i]
    test_x = boston.data[i:]
    train_y = boston.target[:i]
    test_y = boston.target[i:]
    
    reg = FastRGFRegressor()
    try:
        reg.fit(train_x, train_y)
        print("{0} - OK".format(i))
    except:
        print("{0} - Fail".format(i))
