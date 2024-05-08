from tqdm import tqdm

import utils

eval_corpus_path = "birth_dev.tsv"
eval = open(eval_corpus_path, "r")
len_eval = len(eval.readlines())
predictions = ["London"] * len_eval
total, correct = utils.evaluate_places(eval_corpus_path, predictions)
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
else:
    print('No targets provided!')