#Lefki Meidi
#L3 INF G2


import os
import pandas as pd, numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from purity import purity_score


def load_data(csv_file):
    """Load data and return a dataframe with "text" 
    and "label" columns
    
    Args:
        csv_file: path to the input csv file
    """
    # data_df.attrs["descriptions"] = {
    #     ""
    # }

    # TODO: load csv
    data_df = pd.read_csv(csv_file, sep=",")

    # TODO: print data type of each column
    print()

    # TODO: remove "title" column
    data_df.drop(columns=["title"])

    return data_df
    

def sample(data_df: pd.DataFrame):
    """Sample 700 texts for each label

    Args:
        data_df: data frame with "label" and "text" columns
    """
    # TODO: print counts by label
    print("Distribution:")
    print(data_df.value_counts("label"))

    # TODO: sample 700 texts for each label
    sample_df = data_df.groupby("label").sample(700)

    # TODO: print counts by label after downsampling
    print("Distribution:")
    print(data_df.value_counts("label"))
    return sample_df    


def preprocess(text_data, option):
    """Preprocess textual data to have numerical vectors

    Args:
        text_data: input textual data as a data frame
        option: type of encoder to transform categorical data
    """
    if option == "BOW":
        # TODO: use CountVectorizer to have one-hot encoding to represent bag-of-words
        vectorizor = CountVectorizer()
        fitted = vectorizor.fit(text_data)
        data = vectorizor.transform(text_data).toarray()
    elif option == "thenlper/gte-small":
        # TODO: use SentenceTransformer to have embedding vectors
        vectorizor = SentenceTransformer("thenlper/gte-small")
        data = vectorizor.encode(text_data)
    else:
        raise NotImplementedError

    return data

def test_preprocess(option):
    """Test preprocessing methods on similar and dissimilar English sentences.

    Args:
        option: type of encoder to transform categorical data
    """
    print('> Test du prétraitement avec méthode:', option)
    
    data = [
        "She had welcomed my unexpected return as a blessing from heaven.",
        "She had welcomed my return.",
        "She had perceived my unanticipated return as a divine intervention.",
        "She was very happy to see me come back.",
        "She hates my cat, which is unexpected.",
        "She drinks beers, which is a bad habit."
    ]
    
    # Prétraitement des phrases
    prep_data = preprocess(data, option)
    
    # Affichez la phrase de référence
    print(">>> Phrase de référence :", data[0])
    
    # Calculez et affichez les scores de similarité avec les autres phrases
    for i in range(1, len(data)):
        similarity = np.dot(prep_data[0], prep_data[i]) / (norm(prep_data[0]) * norm(prep_data[i]))
        print(f">>> Similarité (échelle de 0 à 1) avec: \"{data[i]}\"")
        print(f">>>\t Score de similarité : {similarity:.4f}")
        print()

# Ajoutez des tests pour comparer les deux méthodes
print("\n=== Test avec Bag-of-Words (BOW) ===")
test_preprocess("BOW")

print("\n=== Test avec Transformer (thenlper/gte-small) ===")
test_preprocess("thenlper/gte-small")


def cluster(data, nb_cluster, current_iter):
    """Cluster data with k-means
    Return a pair of two objects:
    - predicted labels as a list
    - last trained kmeans object

    Args:
        data: numerical data as a list
        nb_cluster: number of clusters
        current_iter: current iteration
    """
    # TODO: create KMeans object
    kmeans = KMeans(n_clusters=nb_cluster, random_state=42, max_iter=current_iter, init='random', n_init=1)

    # TODO: predict labels
    # vectorize comments into bags of words
    preds_label = kmeans.fit_predict(data)
    return (preds_label, kmeans)


def reduce_2D(data):
    """Reduce dimensions (= columns) of numerical data using PCA to only 
    keep first 2 dimensions (the most informative ones)
    Return a pair of two objects:
    - transformed data with only 2 dimensions
    - trained PCA object

    Args:
        data: input data as array
    """
    # TODO: initialize PCA
    method = PCA(n_components=2)
    # TODO: apply PCA
    data_2d = method.fit_transform(data)

    return (data_2d, method)


def scatter_plot(data_2d, ref_labels, pred_labels, pca, kmeans, file_name):
    """Make scatter plot where reference labels are associated with
    markers and predicted labels with colors
    Save figure as a png file.

    Args:
        data_2d: numerical data with 2 dimensions (columns)
        ref_labels: reference labels as a list for each instance
        pred_labels: labels predicted by kmeans for each instance
        pca: trained PCA object (None if kmeans was directly learned)
        from 2D data)
        kmeans: last trained kmeans object
        file_name: name of the output png file
    """
    # Adaptation from:
    # https://scikit-learn.org/dev/auto_examples/cluster/plot_kmeans_digits.html

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = data_2d[:, 0].min(), data_2d[:, 0].max()
    y_min, y_max = data_2d[:, 1].min(), data_2d[:, 1].max()
    x_diff = x_max-x_min
    y_diff = y_max-y_min
    x_min, x_max = x_min-h*10*x_diff, x_max+h*10*x_diff
    y_min, y_max = y_min-h*10*y_diff, y_max+h*10*y_diff
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h*x_diff), np.arange(y_min, y_max, h*y_diff))

    # Obtain labels for each point in mesh. Use last trained model.
    if pca == None:
        try:
            Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel() ])
        except ValueError as e:
            # KMeans.predict() raises a ValueError: Buffer dtype mismatch
            Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel() ].astype(np.float32))
            pass
    else:
        inverse_XXYY= (np.c_[xx.ravel(), yy.ravel() ] @ pca.components_) + pca.mean_ # revert to original space
        try:
            Z = kmeans.predict(inverse_XXYY)
        except ValueError as e:
            Z = kmeans.predict(inverse_XXYY.astype(np.float32))
            pass

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Pastel2,
        aspect="auto",
        origin="lower",
    )

    if pca== None:
        centroids = kmeans.cluster_centers_
    else:
        centroids = pca.transform(kmeans.cluster_centers_)
    
    # plot instances
    MARKERS = ["+","o"]
    ar_ref_labels = np.array(ref_labels)
    ar_pred_labels = np.array(pred_labels)
    # for each ref label
    for n in [0,1]:
        plt.scatter(
            data_2d[:,0][ar_ref_labels==n],
            data_2d[:,1][ar_ref_labels==n],
            c=ar_pred_labels[ar_ref_labels==n],
            cmap=plt.cm.Dark2,
        marker=MARKERS[n])

    # Plot the centroids as a white X
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the textual dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(file_name)
    plt.clf()


def scores_plot(data_lines):
    """Plot purity lines for each tested configuration and
    save figure in "./plots/plot_purity.png"

    Args:
        data_lines: list of triplets for each configuration
        (list of purity scores, list of iterations, label)
    """

    for purity_data, iter_data, label in data_lines:
        plt.plot(iter_data, purity_data, label=label)

    plt.xlabel('Iterations')
    plt.ylabel('Purity Score')
    plt.legend(loc='center')
    plt.title('Purity Score Across Iterations and Configurations')
    plt.savefig("./plots/plot_purity.png")
    plt.clf()
    
    plt.clf()



sample_df = sample(load_data("./data/Test.csv"))

# TODO: get reference labels as a list
ref_labels = sample_df["label"].tolist()

# store purity and iteration number for each tested configuraton
purity_config_lst = []

# for PREPROCESS_NAME in ["BOW"]:
for PREPROCESS_NAME in ["BOW", "thenlper/gte-small"]:
    print("#"*25)
    print(f">>> Prétraitement: {PREPROCESS_NAME} <<<")
    print("#"*25)

    # test_preprocess(PREPROCESS_NAME)

    model_name_short = PREPROCESS_NAME.replace('/','-')
    SUB_DIR = f"./plots/{model_name_short}"
    os.makedirs(SUB_DIR, exist_ok=True)

    text_data = preprocess(sample_df["text"].tolist(), PREPROCESS_NAME)
    text_data_2D, pca = reduce_2D(text_data)

    for NBR_CLUSTER in [2,4]:
        print("> Nombre de clusters:", NBR_CLUSTER)
        
        purity_data, purity_data_2D, iter_data = [], [], []

        for current_iter in [1,2,4,6,8,10]:
            print("> Itération n°", current_iter)
            
            # TODO: make predictions for all dimensions, then (exercise 5) for first 2D
            (labels, kmeans) = cluster(text_data, NBR_CLUSTER, current_iter)
            (labels_2D, kmeans_2D) = cluster(text_data_2D, NBR_CLUSTER, current_iter)

            # TODO: compute purities from predictions
            purity_data.append(purity_score(ref_labels, labels))
            purity_data_2D.append(purity_score(ref_labels, labels_2D))
            iter_data.append(current_iter)

            # TODO: scatter plot for predictions made from data for all dimensions, then for first 2D
            img_model =  f"{SUB_DIR}/plot_MODEL={model_name_short}_N={NBR_CLUSTER}_STEP={current_iter}.png"
            scatter_plot(text_data_2D, ref_labels, labels, pca, kmeans, img_model)   
            img_model_2D =  f"{SUB_DIR}/plot_MODEL={model_name_short}2D_N={NBR_CLUSTER}_STEP={current_iter}_2D.png"
            scatter_plot(text_data_2D, ref_labels, labels_2D, pca, kmeans, img_model)   

        purity_config_lst.append((purity_data, iter_data, f"{model_name_short}_N={NBR_CLUSTER}"))
        purity_config_lst.append((purity_data_2D, iter_data, f"{model_name_short}2D_N={NBR_CLUSTER}"))

scores_plot(purity_config_lst)