# Sentiment analysis using LLM

This program is for sentiment analysis and classification on a given dataset. The given text should be in .csv file and stored line by line. Three sentiment classification will be given: positive, negative and neutral. 

## Prerequisites package:
- openai
- pandas
- json
- concurrent.futures
- retrying
- argparse

Make sure you have those packages ready before running the script. You can install them in the requirements.txt or by this terminal command:

```
pip install openai pandas json concurrent retrying argparse
```

## Specify your OpenAI API key and test file path before running!!!
Please also specify your OpenAI API key and the path csv file of testing data in the run.sh. The default of the testing file is the one in current repository.

## Run Script
To run the script, use the following commend:

```
./run.sh
```

If permission got denied, type the following command and try running it again:


```
chmod +x run.sh
```

## Output
The original text and the output will be stored one by one in the JSON file called "sentiment_classification_result.json".


Feel free to contact jundong0526@gmail.com if you have any issue.