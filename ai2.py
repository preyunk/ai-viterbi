#Importing libraries
import numpy as np
import pandas as pd
import pprint, time
import random


count = 0
train_set = []
with open("train.txt") as f:
    for line in f:
        train_set.append(line.split(' '))
        count += 1

print("Raw Trainset :-")
print(*train_set[:50], sep = '\n')


count = 0
test_set = []
with open("test.txt") as f:
    for line in f:
        test_set.append(line.split(' '))
        count += 1


#into sentences for transition probability
mod_train_set = []
sen = []
for line in train_set:
    if len(line) > 1 and line[0] != '.':
        sen.append((line[0], line[1]))
    if line[0] == '.':
        sen.append((line[0], line[1]))
        mod_train_set.append(sen)
        sen = []

print("\n\n")
print("\n\nModified Train Set :-")
print(*mod_train_set[:5], sep = '\n\n\n')


#into words only for emission probability
train_tag_words = []
for sentence in mod_train_set:
    for w in sentence:
        train_tag_words.append(w)


print("\n\nTrain tagged words :-")
print(train_tag_words[:100])

# tokens
tokens = []
for pair in train_tag_words:
    tokens.append(pair[0])

print("\n\nTokens :-")
print(tokens[:100])

# tags
tags = []
for pair in train_tag_words:
    tags.append(pair[1])

print("\n\nTags :-")
print(tags[:100])


#-----------------------------

# vocabulary
V = set(tokens)


#no. of tags
T = set(tags)

#Probability of a word given a tag)HIDDEN STATE

# computing P(w/t) and storing in T x V matrix
t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))



# compute word given tag: Emission Probability
def word_tag(word, tag, train_bag = train_tag_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    w_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_tag = len(w_tag_list)
    return (count_w_tag, count_tag)



# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability
def t2_given_t1(t2, t1, train_bag = train_tag_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# creating t x t transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)
tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)):
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]

# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))
print(tags_df)

#Viterbi Algo
def Viterbi(words, train_bag = train_tag_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        prob = []
        for tag in T:
            if key == 0:
                t_prob = tags_df.loc['.', tag]
            else:
                t_prob = tags_df.loc[state[-1], tag]
            #compute emission probability
            e_prob = word_tag(words[key], tag)[0]/word_tag(words[key], tag)[1]
            state_prob = e_prob *  t_prob
            prob.append(state_prob)
        prob_max = max(prob)

        max_state = T[prob.index(prob_max)]
        state.append(max_state)
    return list(zip(words, state))


#modified_test_set
mod_test_set = []

sen = []
for line in test_set:
    if len(line) > 1 and line[0] != '.':
        sen.append((line[0], line[1]))
    if line[0] == '.':
        sen.append((line[0], line[1]))
        mod_test_set.append(sen)
        sen = []

random.seed(111)
inp = []
for i in range(5):
    inp.append(random.randint(1,len(mod_test_set)))

print("\n\nSelected Random Integers :-")
print(inp)

test_sent = [mod_test_set[i] for i in inp]

print("\n\n")
print("Picked Test Sentences")
print(*test_sent, sep = '\n\n')

test_tag = [tup for s in test_sent for tup in s]
test_untag = [tup[0] for s in test_sent for tup in s]


print("\nExecuting Viterbi...")
start = time.time()
final = Viterbi(test_untag)
end = time.time()
diff = end-start

print(final)
print("Time taken by viterbi algo is", diff)

correct_count = []

for i,j in zip(final, test_tag):
    if i == j:
        correct_count.append(i)

accuracy = len(correct_count)/len(test_tag)

print("Accuracy on test dataset is ",accuracy)