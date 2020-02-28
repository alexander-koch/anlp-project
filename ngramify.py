import util
from tqdm import tqdm

def ngramify(song, buffer_length, word2idx):
    tokens = song
    for i in range(0, len(song)):
        if i+buffer_length+1 >= len(tokens):
            continue
            
        xs = tokens[i:i+buffer_length]
        y = tokens[i+buffer_length]
        discard = False
        for x in xs:
            if x not in word2idx:
                discard = True
                break
        if discard or y not in word2idx:
            continue
            
        yield xs, y

def main():
    SEQ_LEN = 6
    VOCAB_PATH = "small_vocab.pkl"
    OUTPUT_PATH = "ngrams_small.txt"

    songs = None
    with open("data/sentences.txt", "r") as f:
        songs = [line.rstrip().split(" ") for line in f]

    words = util.load_vocab(VOCAB_PATH)
    word2idx = { word:i for i,word in enumerate(words) }

    with open(OUTPUT_PATH, "w") as f:
        for song in tqdm(songs):
            for xs, y in ngramify(song, SEQ_LEN, word2idx):
                xs = [word2idx[x] for x in xs]
                y = word2idx[y]
                for x in xs:
                    f.write(str(x)+" ")
                f.write(str(y) + "\n")

if __name__ == '__main__':
    main()