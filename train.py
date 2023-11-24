import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
from sklearn import svm
from conversion import convert_jax
import jax.numpy as jnp
import jax

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
for class_idx1 in range(5):
    for class_idx2 in range(5 - class_idx1):
        classes_lt.append(class_idx1 + class_idx2 + 1)
        classes_gt.append(class_idx1)
classes_lt = jnp.array(classes_lt, dtype=np.int32)
classes_gt = jnp.array(classes_gt, dtype=np.int32)

def pred(X):
    y = jnp.dot(X, clf.coef_.T) + clf.intercept_
    support_vector_preds = jnp.where(y < 0, classes_lt, classes_gt)
    class_pred_count = jax.vmap(lambda ix: jnp.count_nonzero(support_vector_preds == ix))(jnp.array(range(6)))
    return jnp.zeros((6), dtype=np.float32).at[jnp.argmax(class_pred_count)].set(1.0)

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

print('Converting model...')
convert_jax(X_train.shape[1:], pred, os.path.join(args.out_directory, 'model.tflite'))
print('Converting model OK')
print('')
