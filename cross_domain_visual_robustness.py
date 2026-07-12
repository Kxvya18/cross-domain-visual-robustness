import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import color
from skimage.feature import hog
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import torchvision
import torchvision.transforms as transforms

OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def extract_hog_features(images, resize_to=(32, 32)):
    features = []
    for image in images:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
            image = color.rgb2gray(image)
        elif image.ndim == 3 and image.shape[-1] == 3:
            image = color.rgb2gray(image)
        image = resize(image, resize_to, anti_aliasing=True)
        features.append(hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, feature_vector=True))
    return np.asarray(features)


def flatten_pixels(images, resize_to=(32, 32)):
    features = []
    for image in images:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
            image = color.rgb2gray(image)
        elif image.ndim == 3 and image.shape[-1] == 3:
            image = color.rgb2gray(image)
        image = resize(image, resize_to, anti_aliasing=True)
        features.append(image.flatten())
    return np.asarray(features)


def load_datasets(data_dir, train_limit, test_limit, download):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
    svhn_train = torchvision.datasets.SVHN(root=data_dir, split="train", download=download, transform=transform)
    svhn_test = torchvision.datasets.SVHN(root=data_dir, split="test", download=download, transform=transform)

    mnist_train_images = mnist_train.data.numpy()[:train_limit]
    mnist_train_labels = mnist_train.targets.numpy()[:train_limit]
    mnist_test_images = mnist_test.data.numpy()[:test_limit]
    mnist_test_labels = mnist_test.targets.numpy()[:test_limit]
    svhn_train_images = svhn_train.data[:train_limit]
    svhn_train_labels = svhn_train.labels[:train_limit]
    svhn_test_images = svhn_test.data[:test_limit]
    svhn_test_labels = svhn_test.labels[:test_limit]

    return {
        "mnist_train_images": mnist_train_images,
        "mnist_train_labels": mnist_train_labels,
        "mnist_test_images": mnist_test_images,
        "mnist_test_labels": mnist_test_labels,
        "svhn_train_images": svhn_train_images,
        "svhn_train_labels": svhn_train_labels,
        "svhn_test_images": svhn_test_images,
        "svhn_test_labels": svhn_test_labels,
    }


def build_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "svm_rbf": SVC(probability=True, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "gaussian_nb": GaussianNB(),
    }


def run_feature_family(feature_name, x_train, y_train, x_in, y_in, x_out, y_out):
    rows = []
    fitted = {}
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        in_pred = model.predict(x_in)
        out_pred = model.predict(x_out)
        in_accuracy = accuracy_score(y_in, in_pred)
        out_accuracy = accuracy_score(y_out, out_pred)
        degradation = ((in_accuracy - out_accuracy) / in_accuracy * 100.0) if in_accuracy else 0.0
        rows.append({
            "feature_family": feature_name,
            "model": model_name,
            "mnist_accuracy": in_accuracy,
            "svhn_accuracy": out_accuracy,
            "relative_degradation_percent": degradation,
        })
        fitted[model_name] = model
    return rows, fitted


def plot_accuracy_transfer(results_df):
    plt.figure(figsize=(10, 5.5))
    melted = results_df.melt(
        id_vars=["feature_family", "model"],
        value_vars=["mnist_accuracy", "svhn_accuracy"],
        var_name="domain",
        value_name="accuracy",
    )
    sns.barplot(data=melted, x="feature_family", y="accuracy", hue="domain")
    plt.title("In-Domain vs Cross-Domain Accuracy")
    plt.xlabel("Feature family")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "domain_transfer_accuracy.png", dpi=180)
    plt.close()


def plot_confusion(y_true, y_pred, title, output_path):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_tsne(features, labels):
    n = min(500, len(features))
    embedding = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto").fit_transform(features[:n])
    plt.figure(figsize=(7, 5.5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels[:n], cmap="tab10", s=12)
    plt.colorbar(label="Digit label")
    plt.title("t-SNE of SVHN HOG Features")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "svhn_hog_tsne.png", dpi=180)
    plt.close()


def plot_hog_distribution(mnist_features, svhn_features):
    plt.figure(figsize=(7, 5))
    plt.hist(mnist_features[0], bins=30, alpha=0.55, label="MNIST HOG sample")
    plt.hist(svhn_features[0], bins=30, alpha=0.55, label="SVHN HOG sample")
    plt.title("HOG Feature Distribution Comparison")
    plt.xlabel("Feature value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hog_feature_distribution.png", dpi=180)
    plt.close()


def adaptation_baselines(x_train_hog, y_train, x_svhn_hog, y_svhn):
    rows = []
    base = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    base.fit(x_train_hog, y_train)
    before = accuracy_score(y_svhn, base.predict(x_svhn_hog))

    adapted = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    adapted.fit(np.vstack([x_train_hog, x_svhn_hog[:200]]), np.hstack([y_train, y_svhn[:200]]))
    after = accuracy_score(y_svhn, adapted.predict(x_svhn_hog))
    rows.append({"baseline": "logistic_regression_few_shot_svhn", "svhn_accuracy": after, "reference_accuracy": before})

    svm = SVC(probability=True, random_state=RANDOM_STATE)
    svm.fit(x_train_hog, y_train)
    ensemble_probs = (base.predict_proba(x_svhn_hog) + svm.predict_proba(x_svhn_hog)) / 2.0
    ensemble_pred = np.argmax(ensemble_probs, axis=1)
    rows.append({"baseline": "logistic_regression_svm_ensemble", "svhn_accuracy": accuracy_score(y_svhn, ensemble_pred), "reference_accuracy": before})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="MNIST-to-SVHN cross-domain visual robustness experiment")
    parser.add_argument("--data-dir", default="data", help="Local dataset directory")
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--download", action="store_true", help="Download MNIST/SVHN into data-dir if missing")
    args = parser.parse_args()

    data = load_datasets(args.data_dir, args.train_limit, args.test_limit, args.download)

    x_train_hog = extract_hog_features(data["mnist_train_images"])
    x_mnist_hog = extract_hog_features(data["mnist_test_images"])
    x_svhn_hog = extract_hog_features(data["svhn_test_images"])

    x_train_raw = flatten_pixels(data["mnist_train_images"])
    x_mnist_raw = flatten_pixels(data["mnist_test_images"])
    x_svhn_raw = flatten_pixels(data["svhn_test_images"])

    pca = PCA(n_components=50, random_state=RANDOM_STATE)
    x_train_pca = pca.fit_transform(x_train_raw)
    x_mnist_pca = pca.transform(x_mnist_raw)
    x_svhn_pca = pca.transform(x_svhn_raw)

    all_rows = []
    fitted_by_feature = {}
    for feature_name, train_x, in_x, out_x in [
        ("hog", x_train_hog, x_mnist_hog, x_svhn_hog),
        ("raw_pixels", x_train_raw, x_mnist_raw, x_svhn_raw),
        ("pca_pixels", x_train_pca, x_mnist_pca, x_svhn_pca),
    ]:
        rows, fitted = run_feature_family(
            feature_name,
            train_x,
            data["mnist_train_labels"],
            in_x,
            data["mnist_test_labels"],
            out_x,
            data["svhn_test_labels"],
        )
        all_rows.extend(rows)
        fitted_by_feature[feature_name] = fitted

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(TABLES_DIR / "cross_domain_accuracy_summary.csv", index=False)
    plot_accuracy_transfer(results_df)

    rf_hog = fitted_by_feature["hog"]["random_forest"]
    y_pred_svhn = rf_hog.predict(x_svhn_hog)
    plot_confusion(data["svhn_test_labels"], y_pred_svhn, "SVHN Confusion Matrix - Random Forest on MNIST HOG", PLOTS_DIR / "svhn_confusion_random_forest_hog.png")
    plot_tsne(x_svhn_hog, data["svhn_test_labels"])
    plot_hog_distribution(x_train_hog, x_svhn_hog)

    adaptation_df = adaptation_baselines(x_train_hog, data["mnist_train_labels"], x_svhn_hog, data["svhn_test_labels"])
    adaptation_df.to_csv(TABLES_DIR / "adaptation_baselines.csv", index=False)

    metadata = pd.DataFrame([{
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }])
    metadata.to_csv(TABLES_DIR / "experiment_metadata.csv", index=False)

    print("Cross-domain visual robustness experiment complete.")
    print(results_df.round(4).to_string(index=False))
    print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
