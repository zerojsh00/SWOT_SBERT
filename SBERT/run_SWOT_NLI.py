import pickle
from sentence_transformers import SentenceTransformer


def main() :

    print("Loading Models")

    MODEL_PATH = './models/BEST_MODEL'
    embedder = SentenceTransformer(MODEL_PATH)

    with open("./models/clustering_model.pkl", "rb") as f:
        clustering_model = pickle.load(f)

    with open("./models/cluster_to_label.pkl", "rb") as g:
        cluster_to_label = pickle.load(g)

    input_str = ""
    while input_str != "종료" :
        input_str = input("SWOT 분석 할 문장 입력, 무한 SWOT 굴레에서 벗어나려면 '종료 입력' : ")
        sentence_embedding = embedder.encode(input_str)

        y_pred_ = clustering_model.predict(sentence_embedding.reshape(1, -1))
        y_pred = [cluster_to_label[c] for c in y_pred_]
        print("{} \t {}".format(y_pred, input_str), end='\n\n')


if __name__ == '__main__':
    main()
