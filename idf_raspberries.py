import sys
sys.path.append('/media/root/Anders J/Pi_Files/py_libraries')


# idf -- the lower the score, the more frequent its usages across documents
# the higher the score, the less frequent its usages
# works on the whole 3GB dataset
import numpy as np
from pygments.lexers import q
from pyspark.ml.feature import Tokenizer, IDF, CountVectorizer, StopWordsRemover, VectorAssembler
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import time
# TODO: define master and uncomment the usb path

start = time.time()
spark = SparkSession.builder \
    .appName("Read JSON with Spark") \
    .config("spark.executor.cores", "4") \
    .master("spark://Pi01:7077") \
    .getOrCreate()

# df = spark.read.json('/Users/sgipson/Desktop/arxiv-metadata-oai-snapshot.json')
df = spark.read.json("/media/raspberries/EXTERNDV/Pi_Files/titles-20MB.json")
df.printSchema()
df.show()
print(df.columns)

# get all titles
titles_df = df.select("title")
#num_entries = 2000000
#titles_df = spark.createDataFrame(all_titles_df.head(num_entries))
# titles_df.coalesce(1).write.json('/Users/sgipson/Desktop/arxiv-metadata-oai-snapshot-all-titles.json')
#print('Sample json file created.')

# tokenize titles into words
tokenizer = Tokenizer(inputCol="title", outputCol="words")
words_data = tokenizer.transform(titles_df)
words_data.show()

# do not want to consider of,and,the etc as being possible keywords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_data = remover.transform(words_data)

# get the vocabulary of the dataset
cv = CountVectorizer(inputCol="filtered_words", outputCol="cv_features")
tf_model = cv.fit(filtered_data)
tf_model.setInputCol("filtered_words")
cv_data = tf_model.transform(filtered_data) # creates new df of data, where vectors are sparse(not including 0s)
vocab = tf_model.vocabulary
indices = {word: index for index, word in enumerate(vocab)}

count = 0
for word, index in indices.items():
    if count < 20:
        count+=1
        print(f"{word}: {index}")


# idf from filtered data
idf = IDF(inputCol="cv_features", outputCol="idf_features")
idf_model = idf.fit(cv_data)
idf_model.setInputCol("cv_features")
idf_df = idf_model.transform(cv_data)  # new df created including the computed idf values

idf_df.select('filtered_words').show(truncate=False)
idf_df.select('cv_features').show(truncate=False)
idf_df.select('idf_features').show(truncate=False)

# link vocab word and idf score
vocab_idf = {}
idfs = idf_model.idf.toArray()
vocab_idf_list = list(zip(vocab, idfs))
for word, idf_score in vocab_idf_list:
    # TODO: check if the word is only alphabetical characters
    if word.isalpha():
        vocab_idf[word] = idf_score
sorted_idfs = sorted(vocab_idf.items(), key=lambda x: x[1], reverse=False) # low to high

output = open("title-scores.txt", "a")
for word, idf_score in sorted_idfs:
    output.write(f"{word}: {idf_score}\n")
output.close()

# word cloud for low scores
lows = sorted_idfs[:100]
lows = dict(lows)
print(lows)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(lows)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Low TF-IDF Scores')
plt.savefig('low-score-titles.png')
plt.clf()
# word cloud for high scores
highs = sorted_idfs[-100:]
highs = dict(highs)
print(highs)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(highs)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('High TF-IDF Scores')
plt.savefig('high-score-titles.png')

end = time.time()
print('Execution time in seconds:', end-start)
spark.stop()
