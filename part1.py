#imports
import re
import os
import numpy as np
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.models import Sequential
from nltk.tokenize import word_tokenize
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import random

#open file
def loadPoetry(filename):
    file = list(open(filename, "r"))
    for word in range(len(file)):
        file[word] = file[word].lower()
    return file

poems = loadPoetry("wwhitman-clean-processed.txt")

#tokenize and add "_EOP_" / "_EOL_"

def tokenize(poems):
    for line in range(len(poems)):
        poems[line] = word_tokenize(poems[line])
    for item in poems:
        if len(item) is 0:
            item.append("_EOP_")
        else:
            item.append("_EOL_")
    flat_list = [item for sublist in poems for item in sublist]
    return flat_list

flat_list = tokenize(poems)
uniqueWords = list(set(flat_list))
flatListNpArray = np.array(flat_list)

# integer encode and 1-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(flatListNpArray)
encodedPoems = to_categorical(integer_encoded)

# making dictionaries
dictOfUnique = {}
dictOfEncoded = {}
for word in range(len(integer_encoded)):
    dictOfUnique[integer_encoded[word]] = flat_list[word]
    dictOfEncoded[integer_encoded[word]] = encodedPoems[word]

# Now we have all the poems shuffling them
np.random.shuffle(encodedPoems)

# Making sequences
seqlen=4
def sequenceX(encodedPoems, seqlen):
    data = []
    labels = []
    for i in range(len(encodedPoems)):
        if (i + seqlen + 1 == len(encodedPoems)):
            break
        temp = []
        for j in range(seqlen + 1):
            if (seqlen is j):
                labels.append(encodedPoems[i + j])
            else:
                temp.append(encodedPoems[i + j])
        data.append(temp)
    return data, labels

x, y = sequenceX(encodedPoems, seqlen)

#Numpy arrays
x_arr = np.array(x)
y_arr = np.array(y)
#building/loading the model
epochs = 50
batch = 64
file = os.path.isfile("Task1Model.h5")
if not file:
    model = Sequential()
    model.add(LSTM(80, input_shape=(x_arr.shape[1], x_arr.shape[2]), return_sequences=False))
    model.add(Dense(x_arr.shape[2], activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_arr, y_arr, epochs=epochs, batch_size=batch)
    model.save("Task1Model.h5")
else:
    model = load_model("Task1Model.h5")

my_dict2 = {y:x for x,y in dictOfUnique.items()}

def getnextword(wordlist):
  intlist=[]
  for each in wordlist:
    if each in my_dict2:
        intlist.append(my_dict2[each])
    else:
        key =random.choice(list(my_dict2))
        intlist.append(my_dict2[key])
  intlist=(np.array(intlist))
  intlist=to_categorical(intlist, len(uniqueWords))
  intlist=intlist[np.newaxis, :]

  p=model.predict(intlist)
  p=p.argmax()
  p=uniqueWords[p]
  return p

def generatepoem(wordlist1):
    finalPoem = []
    wordlist = []
    for i in wordlist1:
        wordlist.append(i.lower())
    x = list(divide_chunks(wordlist, seqlen))
    for i in x:
        while len(i) < seqlen:
            i.append("")
    for i in x:
        nextword = getnextword(i)
        finalPoem.append(nextword)
    return finalPoem

# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


#calling poem generator with length and our words
# to generate a 100 words poem we need 400 words.
text = "And no one here but hens blowing about. If he could sell the place, but then, he can't:" \
       "No one will ever live on it again. " \
       "It's too run down. This is the last of it. " \
       "What I think he will do, is let things smash. " \
       "He'll sort of swear the time away. He's awful! " \
       "I never saw a man let family troubles " \
       "Make so much difference in his man's affairs. " \
       "He's just dropped everything. He's like a child. " \
       "I blame his being brought up by his mother. " \
       "He's got hay down that's been rained on three times. " \
       "He hoed a little yesterday for me: " \
       "I thought the growing things would do him good. " \
       "Something went wrong. I saw him throw the hoe " \
       "Sky-high with both hands. I can see it now-" \
       "Come here- I'll show you- in that apple tree. " \
       "That's no way for a man to do at his age: " \
       "He's fifty-five, you know, if he's a day.' " \
       "Aren't you afraid of him? What's that gun for?' " \
       "Oh, that's been there for hawks since chicken-time. " \
       "John Hall touch me! Not if he knows his friends. " \
       "I'll say that for him, John's no threatener " \
       "Like some men folk. No one's afraid of him; " \
       "All is, he's made up his mind not to stand" \
       "What he has got to stand.' " \
       "'Where is Estelle? " \
       "Couldn't one talk to her? What does she say? " \
       "You say you don't know where she is.' " \
       "'Nor want to! " \
       "She thinks if it was bad to live with him, " \
       "It must be right to leave him.' " \
       "Which is wrong!' " \
       "Yes, but he should have married her.' " \
       "I know.'" \
       "The strain's been too much for her all these years: " \
       "I can't explain it any other way. " \
       "It's different with a man, at least with John: " \
       "He knows he's kinder than the run of men. " \
       "Better than married ought to be as good " \
       "As married- that's what he has always said. " \
       "I know the way he's felt- but all the same! " \
       "I wonder why he doesn't marry her " \
       "And end it. Too late now: she wouldn't have him. " \
       "He's given her time to think of something else. " \
       "That's his mistake. The dear knows my"
words = re.sub("[^\w]", " ",  text).split()
poemg = generatepoem(words)
print(poemg)
