import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
from src.activations import Sigmoid, ReLU, Identity
from src.utils import label_to_onehot, onehot_to_label

from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # Split into 80% train and 20% validation
        N_train = int(0.8 * train_features.shape[0])
        
        # Use random permutation for split
        indices = np.arange(train_features.shape[0])
        np.random.shuffle(indices)
        
        train_idx = indices[:N_train]
        val_idx = indices[N_train:]
        
        # Validation set overwrites test set
        test_features = train_features[val_idx]
        test_labels_reg = train_labels_reg[val_idx]
        test_labels_classif = train_labels_classif[val_idx]
        
        # Keep only train features/labels
        train_features = train_features[train_idx]
        train_labels_reg = train_labels_reg[train_idx]
        train_labels_classif = train_labels_classif[train_idx]

    # Z-score normalization
    means = np.mean(train_features, axis=0, keepdims=True)
    stds = np.std(train_features, axis=0, keepdims=True)
    stds[stds == 0] = 1.0 # Avoid division by zero
    
    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        D = train_features.shape[1] # Number of features (13)
        if args.task == "classification":
            dimensions = (D, args.hidden_dim, 3)
            activations = (ReLU, Sigmoid)
        elif args.task == "regression":
            dimensions = (D, args.hidden_dim, 1)
            activations = (ReLU, Identity)
        else:
            raise ValueError(f"Unknown task: {args.task}")
        method_obj = MLP(dimensions=dimensions, activations=activations)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        if args.method == "dummy_classifier":
            method_obj.fit(train_features, train_labels_classif)
            preds = method_obj.predict(test_features)
        elif args.method == "kmeans":
            method_obj.fit(train_features, train_labels_classif)
            preds = method_obj.predict(test_features)
        elif args.method == "mlp":
            # One-hot encoding of labels
            train_labels_onehot = label_to_onehot(train_labels_classif, C=3)
            from src.losses import CrossEntropy
            
            method_obj.fit(
                train_features,
                train_labels_onehot,
                loss=CrossEntropy,
                epochs=args.max_iters,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
            
            pred_probs = method_obj.predict(test_features)
            preds = onehot_to_label(pred_probs)
            
        acc = accuracy_fn(preds, test_labels_classif)
        f1 = macrof1_fn(preds, test_labels_classif)
        print(f"Classification performance for {args.method}:")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Macro F1-Score: {f1:.4f}")

    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method"

        if args.method == "dummy_classifier":
            method_obj.fit(train_features, train_labels_reg)
            preds = method_obj.predict(test_features)
        elif args.method == "mlp":
            train_labels_reg_reshaped = train_labels_reg[:, np.newaxis]
            
            method_obj.fit(
                train_features,
                train_labels_reg_reshaped,
                loss=MSE,
                epochs=args.max_iters,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
            
            pred_raw = method_obj.predict(test_features)
            preds = pred_raw.flatten()
            
        mse = mse_fn(preds, test_labels_reg)
        print(f"Regression performance for {args.method}:")
        print(f"  MSE: {mse:.4f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=20,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=300,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="hidden layer dimension for MLP",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size for MLP training",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)