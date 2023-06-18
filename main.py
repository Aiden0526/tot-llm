import openai
import pandas as pd
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from retrying import retry
import argparse

# openai.api_key = "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"

##断点重爬api
@retry(stop_max_attempt_number=3,wait_fixed=2000)
def classify(prompt,api_key):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=.3,
            max_tokens=32,
        )

        return response.choices[0].text.strip()
    except Exception as e:
        print("There is an error: ", e)
        return "neutral"

def thread_task(row, myprompt,api_key):
    text = row['text']
    combined_prompt = f"{myprompt} \n\n###\n\n{text}"
    sentiment = classify(combined_prompt,api_key)
    return {
        "text": text,
        "sentiment": postprocess(sentiment)
    }


def sentiment_classification(test_file_path,myprompt,api_key,n_thread = 10):

    # Load the test data
    test_df = pd.read_csv(test_file_path,encoding='latin-1')

    sentiment_results = []

    ## run the thread task under thread pool
    with ThreadPoolExecutor(max_workers=n_thread) as e:
        futures = {e.submit(thread_task, row, myprompt,api_key): row for _, row in test_df.iterrows()}
        for future in as_completed(futures):
            myresult = future.result()
            if myresult is not None:
                sentiment_results.append(myresult)

        # Save the results to a JSON file
    output_file = os.path.join(os.getcwd(),"sentiment_classification_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(sentiment_results, f,indent=4)



    print("Sentiment classification completed. Results are saved to sentiment_classification_results.json")

## this function is to check whether one of the three sentiments is in the sentiment generated from the api to determine the final sentiment saved in the json file.
def postprocess(sentiment):
    sentiment = sentiment.lower()

    # The valid sentiments
    expected_sentiments = ["positive", "negative", "neutral"]

    # Check if each valid sentiment is in the output
    for expected_sentiment in expected_sentiments:
        if expected_sentiment in sentiment:
            return expected_sentiment

    # If no valid sentiment is found, return a default
    return "neutral"



def parse_arge():
    args = argparse.ArgumentParser()
    args.add_argument('--api_key',required=True,help='Enter your OpenAi API key')
    args.add_argument('--test_file',required=True,help='Enter the test file path that you want to test')

    args = args.parse_args()
    return args



if __name__ == '__main__':
    args = parse_arge()
    # Specify the api key and paths of testing CSV files
    api_key = args.api_key
    test_file_path = args.test_file

    prompt_file = 'prompt.txt'
    with open(prompt_file,'r') as f:
        myprompt = f.read()


    # run the classification function
    sentiment_classification(test_file_path, myprompt,api_key)