import spacy

# All code snippets have been taken from the assignment brief,
# with the exception of the last, and as such will not be commented.
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# I find it interesting that the similarity between cat, monkey and banana
# takes into account common associations with these words. Rather than just 
# looking at the words themselves, it looks at the context in which they are
# used.

new_word1 = nlp("car")
new_word2 = nlp("park")
new_word3 = nlp("garage")
print(new_word1.similarity(new_word2))
print(new_word3.similarity(new_word2))
print(new_word3.similarity(new_word1))

# If the simpler en_core_web_sm language model is used, the similarity between the words is
# much less accurate. Spacy warns that the model does not contain word vectors
# and so the similarity is not reliable.