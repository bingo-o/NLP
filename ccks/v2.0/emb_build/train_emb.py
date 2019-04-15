import word2vec

paths = ["train_vec.txt"]
sizes = [300]


def tran(emb_path):
    print(emb_path)
    model = word2vec.load(emb_path)
    vocab, vectors = model.vocab, model.vectors
    print("shape of embeddings: {}".format(vectors.shape))

    new_path = emb_path.split(".")[0] + ".txt"
    print("generating embeddings...")
    with open(new_path, "w") as f:
        for word, vector in zip(vocab, vectors):
            f.write(word + "\t" + " ".join(map(str, vector)) + "\n")
    print("completed!")


for path in paths:
    for size in sizes:
        emb_path = path.split(".")[0].split("_")[1] + "_" + str(size) + ".bin"
        word2vec.word2vec(path, emb_path, min_count=5, size=size, verbose=True)
        tran(emb_path)
