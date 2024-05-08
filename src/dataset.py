import random
import torch
from torch.utils.data import Dataset
import argparse

"""
The input-output pairs (x, y) of the NameDataset are of the following form:

  x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

Note that the NameDataset should take the pretraining_dataset defined in run.py
as an input. This is to allow the vocab specification of the NameDataset to be
the same as that of the pretraining dataset.

"""

class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad
        self.itos = pretraining_dataset.itos # tok2id dictionary
        self.stoi = pretraining_dataset.stoi #id2tok dictionary
        self.block_size = pretraining_dataset.block_size # context size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))
        '''
        self.data  =
        ['Where was John Molyneux born?\tWarrington', 'Where was Maggie Fitzgibbon born?\tMelbourne', 'Where was Bruce Metcalf born?\tAmherst', 'Where was Janet Ramsey Johnson born?\tAdelaide', 'Where was Ien Ang born?\tJava', '']
        '''

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        
        '''
        this returns x, y 
        x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
        y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
        
        We use teacher forcing in causal fashion and so the expected next token is the input at the previous step.   
        '''
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x)) # pads with empty square to make up to the context length
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
        x = x[:-1] # it's of length block_size-1 as the last character is the output
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long) # token to id
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long) # token to id
        return x, y


"""
The vocabulary is to be accessible via two dictionaries:
  self.stoi: a dictionary from characters in the vocabulary to indices of type
      int
  self.itos: a dictionary from indices of type int to characters in the
      vocabulary

The vocabulary has the following form: 

  Identifier 0 must be assigned to the unicode element u"\u25A1".
      This is the empty_square_character.
      Further, let self.PAD_CHAR = u"\u25A1"
  Identifier 1 must be assigned to the unicode element u"\u2047".
      This is the doublequestionmark character, which we'll use
      as a sentinel to represent that text is missing from the input
      Further, let self.MASK_CHAR = u"\u2047"
  Identifiers 2, ..., len(self.itos)-1 should be the sorted list of characters
      that appear in the data argument.

--------------
Masking Specification

The __getitem__ function takes an index and returns a data point (x, y) where
x and y are Long tensors of length self.block_size. x encodes the input
sequence, and y encodes the output sequence.

0. Use the idx argument of __getitem__ to retrieve the element of self.data
at the given index - self.data is a list with each element corresponds to a sentence 
in wiki.txt. We'll call the resulting data entry "a document". wiki.txt has multiple 
lines seperated by `\n` and each line has the info about birthplace of a personality.  

1. Randomly truncate the document (a line randomly picked in step 0) to a length no less than 4 characters,
and no more than int(self.block_size*7/8) characters.

- Randomly pick a number between 4 and int(self.block_size*7/8) - truncated_len. 

- Randomly pick a sequence of characters (of length truncated_len) from the document. 
    - if the truncated_len > len(doc) - Use doc[:len(doc)]
    - if len(doc)>=truncated_len - pick a number between 0 and len(doc)-truncated_len+1 (to avoid overflow) and use that starting index. doc[starting_index:starting_index+truncated_length]
    

2. Now, break the (truncated) document into three substrings:
    
    [prefix] [masked_content] [suffix]

  In other words, choose three strings prefix, masked_content and suffix
    such that prefix + masked_content + suffix = [the original document].
  The length of [masked_content] should be random, and 1/4 the length of the
    truncated document on average.
    
 - Pick a random number from 1 to int(truncated_len/4) - dnotes masked length
 - Pick a random number from 0 to truncated_len - int(truncated_len/4) + 1 - masked_content_index (starting index of masked content)
 - prefix = truncated_doc[:masked_content_index]
   suffix = truncated_doc[masked_content_index + masked_content_len:]
   masked_content = truncated_doc[masked_content_index:masked_content_index+masked_content_len]
   

3. Rearrange these substrings into the following form:

    [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
  
  This resulting string, denoted masked_string, serves as the output example.
  Here MASK_CHAR is the masking character defined in Vocabulary Specification,
    and [pads] is a string of repeated PAD_CHAR characters chosen so that the
    entire string is of length self.block_size.
  Intuitively, the [masked_content], a string, is removed from the document and
    replaced with MASK_CHAR (the masking character defined in Vocabulary
    Specification). After the suffix of the string, the MASK_CHAR is seen again,
    followed by the content that was removed, and the padding characters.

4. We now use masked_string to construct the input and output example pair. To
do so, simply take the input string to be masked_string[:-1], and the output
string to be masked_string[1:]. In other words, for each character, the goal is
to predict the next character in the masked string.

5. Making use of the vocabulary that you defined, encode the resulting input
and output strings as Long tensors and return the resulting data point.

----------------
Here are some examples of input-output pairs (x, y):

  x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: Jaco⁇enry ⁇b H□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: aco⁇enry ⁇b H□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□


We train the model to learn to output next token and the answer (masked tokens) between two MASK_TOKEN

"""
class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        # build the vocabulary from pre-training corpus. Also add mask and pad character to the vocab
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data)))) # sort the chracters
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        #add mask and pad character
        chars.insert(0, self.MASK_CHAR) 
        chars.insert(0, self.PAD_CHAR)
        
        # ids are provided based on the sorting order
        self.stoi = { ch:i for i,ch in enumerate(chars) } 
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n') # list of sentences in wiki.txt seperated by `\n`, each line has the info about birth place of famous personalities

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) # total lines in wiki.txt

    def __getitem__(self, idx):
        
#         doc = self.data[idx]
#         truncated_len = int(torch.randint(low=4, high=int(self.block_size*7/8)+1, size=(1,))[0])
#         if truncated_len > len(doc):
#             trucated_doc_index = 0
#         else:
#             trucated_doc_index = int(torch.randint(low=0, high=len(doc)-truncated_len+1, size=(1,))[0])
        
#         truncated_doc = doc[trucated_doc_index:trucated_doc_index+truncated_len]
#         masked_content_len = int(torch.randint(low=1, high=int(truncated_len/4)+1, size=(1,))[0])
#         masked_content_index = int(torch.randint(low=0, high=truncated_len - int(truncated_len/4) + 1, size=(1,))[0])
        
#         prefix = truncated_doc[:masked_content_index]
#         suffix = truncated_doc[masked_content_index + masked_content_len:]
#         masked_content = truncated_doc[masked_content_index:masked_content_index+masked_content_len]
        
#         masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content + self.PAD_CHAR*(self.block_size - len(prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content) + 1)
        
        
#         x = masked_string[:-1]
#         y = masked_string[1:]
#         x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
#         y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
#         return x, y
        
        
        doc = self.data[idx]
        truncated_len = int(torch.randint(low=4, high=int(self.block_size * 7/8) + 1, size=(1,))[0]) #random.randint(4, int(self.block_size*7/8))
        truncated_doc = doc[:truncated_len]

        #masked_content_len = int(random.normalvariate(truncated_len/4, 1))
        masked_content_len = int(torch.randint(low=1, high=2*int(truncated_len/4), size=(1,))[0])
        masked_content_index = int(torch.randint(low=0, high=truncated_len - int(truncated_len/4) + 1, size=(1,))[0])
        
        prefix = truncated_doc[:masked_content_index]
        suffix = truncated_doc[masked_content_index + masked_content_len:]
        masked_content = truncated_doc[masked_content_index : masked_content_index + masked_content_len]

        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content + self.PAD_CHAR*(self.block_size - len(prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content) + 1)

        x = masked_string[:-1]
        y = masked_string[1:]
        
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y
    
    
                            
           
      

"""
Code under here is strictly for your debugging purposes

python src/dataset.py namedata
python src/dataset.py charcorruption
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption"])
    args = argp.parse_args()
    '''
    This prints out 4 samples of x, y from birth_places_train.csv dataset
    
    x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    x: Where was John Stephen born?⁇Glasgow⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    y: □□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Glasgow⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    x: Where was Georgina Willis born?⁇Australia⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Australia⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
    
    '''
    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(open('wiki.txt', encoding='utf-8').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('birth_places_train.tsv', encoding='utf-8').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(open('wiki.txt', encoding='utf-8').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))

