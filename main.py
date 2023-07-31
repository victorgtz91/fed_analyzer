import streamlit as st
import requests
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from transformers import pipeline
from transformers import DistilBertTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from matplotlib.ticker import MaxNLocator

# Add a title and a button to the Streamlit UI
st.title("Welcome, this is the Fed Analyzer")
if st.button("RUN"):

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

# Add to the list of stop words
stop_words.update(["federal", "reserve", "financial", "data", "system", "board", "page", "committee"])

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Specify the model
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

# Initialize the sentiment analysis pipeline with the specified model
nlp_sentiment = pipeline('sentiment-analysis', model=model_name)

# Initialize spacy for name removal
nlp = spacy.load('en_core_web_sm')

# Send GET request to the webpage
response = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")

# Parse the HTML content of the webpage
soup = BeautifulSoup(response.content, 'html.parser')

# https://www.federalreserve.gov/newsevents/pressreleases/monetary20230201a.htm
# Find all links that match the pattern "/pressreleases/monetaryYYYYMMDDa.htm"
links = soup.find_all('a', href=re.compile(r'/pressreleases/monetary\d{8}a\.htm'))

# Extract the dates from the links
dates = [re.search(r'\d{8}', link['href']).group() for link in links]

# Convert the dates to a pandas DataFrame
df_dates = pd.DataFrame(dates, columns=['Date'])

# Create an empty DataFrame to store the results
df_words = pd.DataFrame(
    columns=['Date', 'Word1', 'Word2', 'Word3', 'Word4', 'Word5', 'Sentiment', 'Inflation_Count', 'Market_Count']
)

# Iterate over each date in the DataFrame
for _, row in df_dates.iterrows():
    date = row['Date']

    # URL of the FOMC minutes
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{date}a.htm"

    # Send GET request to the URL
    response = requests.get(url)

    # Parse the webpage content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the text from the webpage
    text = soup.get_text()

    # Use spacy to remove names from the text
    doc = nlp(text)
    text_no_names = " ".join([token.text for token in doc if token.ent_type_ != "PERSON"])

    # Tokenize the text
    sentences = sent_tokenize(text_no_names)

    # Initialize a variable to store the total sentiment score and a count of sentences
    total_sentiment_score = 0
    sentence_count = 0

    # Iterate over each sentence
    for sentence in sentences:
        # Tokenize the sentence
        sentence_tokens = tokenizer.tokenize(sentence)

        # Truncate the sentence to 510 tokens to leave room for special tokens
        if len(sentence_tokens) > 510:
            sentence_tokens = sentence_tokens[:510]

        # Convert the tokens back to a string
        sentence = tokenizer.convert_tokens_to_string(sentence_tokens)

        # Apply the sentiment analyzer to the sentence
        result = nlp_sentiment(sentence)

        # Extract the sentiment label and score from the result
        sentiment_label = result[0]['label']
        sentiment_score = result[0]['score']

        # If the sentiment label is 'NEGATIVE', make the sentiment score negative
        if sentiment_label == 'NEGATIVE':
            sentiment_score = -sentiment_score

        # Add the sentiment score to the total sentiment score
        total_sentiment_score += sentiment_score

        # Increment the sentence count
        sentence_count += 1

    # Calculate the average sentiment score
    average_sentiment_score = total_sentiment_score / sentence_count

    # Tokenize the text, filter out stopwords and non-alphabetic tokens, and count word frequencies
    word_freq = Counter(
        [word.lower() for word in word_tokenize(text_no_names) if word.isalpha() and word.lower() not in stop_words]
    )

    # Get 5 most common words
    common_words = word_freq.most_common(5)

    # Count the occurrences of "inflation" and "market"
    inflation_count = word_freq['inflation']
    market_count = word_freq['market']

    # Add the results to the DataFrame
    new_row = pd.DataFrame(
        {'Date': [date], 'Word1': [common_words[0][0]], 'Word2': [common_words[1][0]], 'Word3': [common_words[2][0]],
         'Word4': [common_words[3][0]], 'Word5': [common_words[4][0]], 'Sentiment': [average_sentiment_score],
         'Inflation_Count': [inflation_count], 'Market_Count': [market_count]})
    df_words = pd.concat([df_words, new_row], ignore_index=True)

# Print the DataFrame
# print(df_words)

# Save the DataFrame to a CSV file
df_words.to_csv('results.csv', index=False)

# Convert 'Date' to datetime
df_words['Date'] = pd.to_datetime(df_words['Date'], format='%Y%m%d')

# Sort DataFrame by date
df_words = df_words.sort_values('Date')

# Save the sorted DataFrame to a CSV file
df_words.to_csv('sorted_results.csv', index=False)

# Load the SP500 data
df_spy = pd.read_csv('spy.csv')

# Convert 'Date' to datetime in df_words
df_words['Date'] = pd.to_datetime(df_words['Date'], format='%Y%m%d')

# Convert 'Date' to datetime in df_spy
df_spy['Date'] = pd.to_datetime(df_spy['Date'])

# Merge the dates
df_merged = pd.merge(df_words, df_spy, on='Date', how='inner')

# Sort DataFrame by date
df_merged.sort_values('Date', inplace=True)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot sentiment scores in the left subplot
ax1.plot(df_merged['Date'], df_merged['Sentiment'])
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment Score')
ax1.set_title('Sentiment Score Over Time')

# Calculate the differences
differences = df_merged['Inflation_Count'] - df_merged['Market_Count']

# Create a color list: 'blue' for positive values, 'red' for negative values
colors = ['b' if value >= 0 else 'r' for value in differences]

# Create the bar plot
bars = ax2.bar(df_merged['Date'], differences, color=colors, width=10)

# Create a secondary y-axis
ax3 = ax2.twinx()

# Plot 'Close' column from spy.csv on the secondary y-axis
line = ax3.plot(df_merged['Date'], df_merged['Close'], color='green')

# Set the y-axis limit for the secondary y-axis
ax3.set_ylim([200, 500])

# Set labels and title
ax2.set_xlabel('Date')
ax2.set_ylabel('Difference Count')
ax3.set_ylabel('Close Price')
ax2.set_title('Difference between Inflation and Market mentions Over Time')

# Set y-axis to only display integer values
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

# Create a legend
ax2.legend([bars, line[0]], ['Market vs Inflation Count', 'SP500 Price Level'])

# Show the figure
#plt.savefig('plot.png', dpi=600)
#plt.show()

# Print the DataFrame
# print(df_words[['Date', 'Inflation_Count']])

    # Show the figure in the Streamlit UI instead of saving it to a file
    st.pyplot(fig)

    # Print the DataFrame in the Streamlit UI
    st.write(df_words[['Date', 'Inflation_Count']])
