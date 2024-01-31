import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
from sklearn import svm
from conversion import convert_jax
import jax.numpy as jnp
import jax
import pickle

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Train custom ML model')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--max-iter', type=int, required=True, default=-1)
parser.add_argument('--out-directory', type=str, required=True)

args, _ = parser.parse_known_args()

out_directory = args.out_directory

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'))
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'))
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

# sparse representation of the labels (1-based)
Y_train = np.argmax(Y_train, axis=1)
Y_test = np.argmax(Y_test, axis=1)

# num_features = X_train.shape[1]
# num_classes = Y_train.shape[0]
#
# print('num features: ' + str(num_features))
# print('num classes: ' + str(num_classes))
# print('mode: ' + str(input.mode))

print('Training linear SVM classifier on', str(X_train.shape[0]), 'inputs...')
clf = svm.SVC(kernel='linear', max_iter=args.max_iter)
clf.fit(X_train, Y_train)

print('Training linear SVM classifier OK')
print('')

print('Mean accuracy (training set):', clf.score(X_train, Y_train))
print('Mean accuracy (validation set):', clf.score(X_test, Y_test))
print('')

# here comes the magic, provide a JAX version of the `proba` function

classes_lt = []
classes_gt = []
for class_idx1 in range(len(clf.classes_) - 1):
    for class_idx2 in range(len(clf.classes_) - 1 - class_idx1):
        l = np.zeros(len(clf.classes_), dtype=np.float32)
        r = np.zeros(len(clf.classes_), dtype=np.float32)
        l[class_idx1 + class_idx2 + 1] = 1
        r[class_idx1] = 1
        classes_lt.append(l)
        classes_gt.append(r)

classes_lt = jnp.array(classes_lt, dtype=jnp.float32)
classes_gt = jnp.array(classes_gt, dtype=jnp.float32)

def pred(X):
    y = jnp.dot(X[0], clf.coef_.T) + clf.intercept_
    l = jnp.where((y < 0.0), jnp.array(0, dtype=jnp.float32), jnp.array(1, dtype=jnp.float32)).reshape((clf.coef_.shape[0], 1))
    r = jnp.where((y >= 0.0), jnp.array(0, dtype=jnp.float32), jnp.array(1, dtype=jnp.float32)).reshape((clf.coef_.shape[0], 1))
    res = (l * classes_gt) + (r * classes_lt)
    return jnp.sum(res, axis=0).reshape((1,len(clf.classes_)))

print(' ')
print('Calculating SVM accuracy...')
num_correct = 0
for idx in range(len(Y_test)):
    p0 = pred(jnp.array(X_test[idx].reshape(1, -1), dtype=jnp.float32))
    print(p0)
    print(jnp.argmax(p0))
    print(Y_test[idx])
    if Y_test[idx] == jnp.argmax(p0):
        num_correct += 1
print(f'Accuracy (validation set): {num_correct / len(Y_test)}')

with open(os.path.join(args.out_directory, 'model.pkl'),'wb') as f:
    pickle.dump(clf,f)

with open(os.path.join(args.out_directory, 'model.pkl'), 'rb') as f:
    clf2 = pickle.load(f)
    print(type(clf2))
    if type(clf2) == svm.SVC:
        print('svm.SVC')

print('Converting model...')
convert_jax(X_train.shape[1:], pred, os.path.join(args.out_directory, 'model.tflite'))
print('Converting model OK')
print('')

print(clf.coef_.shape)
print(len(clf.classes_))
print(str(vars(clf)))
