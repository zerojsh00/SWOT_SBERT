import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report


def define_argparser() :
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--n_companies', type=int, required=True)
    p.add_argument('--output_dir', type=str, default='./output')
    p.add_argument('--train_dataset_dir', type=str, default='../Dataset/SWOT_NLI')
    p.add_argument('--test_dataset_dir', type=str, default='../Dataset/SWOT_quad')

    config = p.parse_args()
    return config

def plot_inertia(SBERT_embeddings_train, model_path, n_companies):
    inertia_list = []
    for num_clusters in range(1, 11):
        clustering_model = KMeans(n_clusters=num_clusters, random_state=50)
        clustering_model.fit(SBERT_embeddings_train)
        inertia_list.append(clustering_model.inertia_)

    # plot the inertia curve
    plt.plot(range(1, 11), inertia_list, 'o--', color='grey')
    plt.scatter(range(1, 11), inertia_list)
    plt.xlabel("Number of Clusters", size=13)
    plt.ylabel("Inertia Value", size=13)
    plt.title("Different Inertia Values for Different Number of Clusters", size=10)
    plt.grid(True)
    plt.savefig('{}/Inertia_{}.png'.format(model_path, str(n_companies).zfill(4)), dpi=300)

def plot_PCA3d(SBERT_embeddings_train, clustering_model, quad_dataset ,model_path, n_companies):
    pca3 = PCA(n_components=3).fit(SBERT_embeddings_train)
    pca3d = pca3.transform(SBERT_embeddings_train)

    data = pd.DataFrame(data=pca3d)
    data["cluster"] = clustering_model.labels_
    data["label"] = quad_dataset['label'].to_list()
    data.columns = ['PC1', 'PC2', 'PC3', 'cluster', 'label']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    # 임베딩 결과를 fitting한 train 데이터셋의 클러스터링 결과를 이용하여 plot 함
    for i in data['cluster'].unique():
        ax.scatter(data[data['cluster'] == i]['PC1'], data[data['cluster'] == i]['PC2'],
                         data[data['cluster'] == i]['PC3'], alpha=0.2)

    plt.legend(data['cluster'].unique())
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.savefig('{}/KMeans_Clusters_{}.png'.format(model_path, str(n_companies).zfill(4)), dpi=300)

    # 임베딩 결과를 실제 레이블을 이용하여 plot 함
    for i in data['label'].unique():
        ax.scatter(data[data['label'] == i]['PC1'], data[data['label'] == i]['PC2'],
                         data[data['label'] == i]['PC3'], alpha=0.2)

    plt.legend(data['label'].unique())
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.savefig('{}/SWOT_data_points_{}.png'.format(model_path, str(n_companies).zfill(4)))


def NLI_2_quad(train_dataset):
    # SWOT NLI 형태의 train 데이터셋을 4개 클래스 데이터셋으로 변환함
    labeled_data = []
    for row in train_dataset.iterrows():
        l1, l2 = row[1][-1:][2]
        labeled_data.append([row[1][0], l1])
        labeled_data.append([row[1][1], l2])

    quad_dataset = pd.DataFrame(labeled_data, columns=["sentence", "label"])
    quad_dataset = quad_dataset.drop_duplicates()
    return quad_dataset

def main(config) :
    N_COMPANIES = config.n_companies

    TRAIN_DATASET_DIR = config.train_dataset_dir
    TRAIN_DATASET_PATH = TRAIN_DATASET_DIR + '/{}/train.tsv'.format(str(N_COMPANIES).zfill(4))

    TEST_DATASET_DIR = config.test_dataset_dir
    TEST_DATASET_PATH = TEST_DATASET_DIR + '/{}/test.tsv'.format(str(N_COMPANIES).zfill(4))

    OUTPUT_DIR = config.output_dir
    MODEL_PATH = OUTPUT_DIR + '/SWOT_SBERT_{}'.format(str(N_COMPANIES).zfill(4))
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformer(MODEL_PATH)

    test_dataset = pd.read_csv(TEST_DATASET_PATH, header=None, sep='\t') # test dataset
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH, header=None, sep='\t') # train dataset

    # 임베딩을 위해 NLI 형태의 train 데이터셋을 이용하여 4개 클래스 데이터셋을 만듦
    quad_dataset = NLI_2_quad(train_dataset)
    # 성능 측정을 위한 test 데이터셋
    quad_dataset_test = pd.DataFrame([[d[0], d[1]] for d in zip(test_dataset[0], test_dataset[1])], columns=["sentence", "label"])

    # SBERT Embeddings
    SBERT_embeddings_train = embedder.encode(quad_dataset['sentence'].to_list())  # SBERT를 활용한 train dataset의 임베딩 벡터
    SBERT_embeddings_test = embedder.encode(quad_dataset_test['sentence'].to_list())  # SBERT를 활용한 train dataset의 임베딩 벡터

    # Plot Inertia
    plot_inertia(SBERT_embeddings_train, MODEL_PATH, N_COMPANIES)

    # K-means 클러스터링
    num_clusters = 4
    clustering_model = KMeans(n_clusters=num_clusters, random_state=50)
    clustering_model.fit(SBERT_embeddings_train)  # train 데이터셋으로 클러스터링 실시

    # plot 3d-PCA
    plot_PCA3d(SBERT_embeddings_train, clustering_model, quad_dataset, MODEL_PATH, N_COMPANIES)

    # NLI train에 사용했던 train 데이터를 4개 클래스로 변환(quad_dataset)하고,
    # quad_dataset을 임베딩하였을 때 만들어진 클러스터마다 가장 많이 나온 실제 레이블을 해당 클러스터의 정답 레이블로 간주함
    vc = pd.DataFrame(data = [clustering_model.labels_, quad_dataset['label']], index=["cluster", "label"]).T.value_counts()
    cluster_to_label = {cluster[0]: cluster[1] for cluster in vc[:4].index} # To Do : 해당 라인은 나이브하게 구현되었으나, 적은 데이터셋에서는 고려해야 할 부분이 많음

    # 이후, test 데이터셋의 군집을 예측했을 때 나온 값과 앞서 간주된 레이블을 비교하여 성능을 측정함
    y_pred_ = clustering_model.predict(SBERT_embeddings_test)
    y_pred = [cluster_to_label[c] for c in y_pred_]
    
    y_true = quad_dataset_test["label"].to_list()
    print(classification_report(y_true, y_pred, output_dict=False))

    # with open('../output_for_paper/SBERT_output/SBERT_report_{}.txt'.format(str(N_COMPANIES).zfill(4)) , "w") as text_file:
    with open(MODEL_PATH+'/SBERT_report_{}.txt'.format(str(N_COMPANIES).zfill(4)) , "w") as text_file:
        print(classification_report(y_true, y_pred, digits=4), file=text_file)

    # report = classification_report(y_true, y_pred, output_dict=True)
    # pd.DataFrame(report).transpose().to_csv(MODEL_PATH+'/report_{}.csv'.format(str(N_COMPANIES).zfill(4)), index = False)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
