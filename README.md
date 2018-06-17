# my_pytorch_chatbot
 pytorch_chatbot(with gensim loaded model)
### Source of the corpus
https://github.com/Marsan-Ma/chat_corpus
### Source code
My codes is modified from the code from https://github.com/ywk991112/pytorch-chatbot
### Training
#### Seq2seq
Train the seq2seq model with pretrained word2vector model.
```
python main.py -tr <CORPUS_FILE_PATH> -pre <PRETRAINED_MODEL_FILE_PATH> -la 1 -hi 300 -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
```
> python main.py -tr ./data/movie_subtitles_en.txt -pre ./save/model/GoogleNews-vectors-negative300.bin -la 1 -hi 300 -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
### Test
Test the seq2seq model
#### test randomly
```
python main.py -te <SEQ2SEQ_MODEL_FILE_PATH> -pre .<PRETRAINED_MODEL_FILE_PATH> -c <TRAINING_CORPUS_FILE_PATH> -cd <TESTING_CORPUS_FILE_PATH>
```
> python main.py -te ./save/model/movie_subtitles_en/1-1_300/26000_backup_bidir_model.tar -pre ./save/model/GoogleNews-vectors-negative300.bin -c ./data/movie_subtitles_en.txt - ./data/movie_subtitles_en.txt

### Loss Graph
Draw the loss graph.
```
python main.py -lo <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -hi 300
```
> python main.py -lo ./save/model/movie_subtitles_en/1-1_300/5_backup_bidir_model.tar -c ./data/movie_subtitles_en.txt -hi 300
