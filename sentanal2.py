# %%


from transformers import pipeline
import pandas as pd

from transformers import TextClassificationPipeline, TFAutoModelForSequenceClassification, AutoTokenizer



MODEL_DIR = 'digitalepidemiologylab/covid-twitter-bert-v2'


# %%

myTable = pd.read_csv("/storage/home/mzm6486/work/DS340W/myTable.csv")

tweets = myTable['text']
tweets = pd.DataFrame(tweets)

tweets = tweets[6000:12000]


# print(tweets.head())

# %%

# classifier = pipeline(task="sentiment-analysis", model='digitalepidemiologylab/covid-twitter-bert-v2')
 
# results = classifier(tweets['text'])


# results = (
#     tweets
#     .assign(sentiment = lambda x: x['text'].apply(lambda s: classifier(s)))
#     .assign(
#          label = lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
#          score = lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
#     )
# )



# %%
# Feature extraction pipeline
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

pipeline = TextClassificationPipeline(model=model,
                                      tokenizer=tokenizer,
                                      framework='tf', 
                                      task="sentiment-analysis",
                                      device=0)

# results = pd.DataFrame()

# for t in tweets:
#     print(t)
myResult = pd.DataFrame(pipeline(list(tweets['text'])))
    # results = pd.concat([results, myResult])


# %%


myResult.to_csv("sentResults6000to12000.csv")


# %%



