# Higher_Education_Chatbot
A chatbot with tkinter gui that answers queries related to higher education using certain datasets and also provides ML prediction of admission chances. Additionally, it is also trained with the chatterbot corpus.

## Prequisites:
- Python environment with python version 3.7 or lesser (currently, chatterbot module is incompatible with version 3.8 or higher)
  - If possible, create and utiise an Anaconda environment with python version 3.7
- Chatterbot Module (Run these exact commands on the command prompt of the specific environment)
  - `pip install chatterbot`
  - `pip install spacy==2.3.5`
  - `python -m spacy download 'en'` - Perform this command in the command prompt of the specific environment with administrator privileges (Right click and run as administrator on the terminal)
  - `pip install chatterbot-corpus`
- pandas Module
  - `pip install pandas`
- tkinter Module
  - `pip install tk`
- ttkthemes Module
  - `pip install ttkthemes`
- matplotlib Module
  - `pip install matplotlib`
- seaborn Module
  - `pip install seaborn`
- sklearn Module
  - `pip install sklearn`
- Download all the files in the folder and keep them in the same workspace
- Run the `Higher_Education_Chatbot.py` file on your python IDE
- A tkinter GUI with a chatbot type interface and a plot button on the right should appear
  - ![Capture](https://user-images.githubusercontent.com/45400093/116785537-ba85cc80-aab7-11eb-93ed-2ef773b6401d.PNG)

## Flowchart:
![WhatsApp Image 2021-04-29 at 4 55 51 PM](https://user-images.githubusercontent.com/45400093/116784572-9bd10700-aab2-11eb-966c-8a61545d4aef.jpeg)

## Working:
- College Admission Prediction:
  - Queries such as 'What are my chances of college admission?', 'What are my odds of getting into a college?', 'Can I get a seat in a top college', etc. enter this flow.
  - Any question with keywords such as 'chance', 'chances', 'get', 'get in', etc. enter this flow.
  - The chatbot subsequently asks for your GRE, TOEFL, SOP, LOR, Desired University Rating (1 being highest and 5 is lowest) and whether you have done any research or not.
  - Finally it predicts your admission chances(using SVR) in percentage and presents a graph of correlation when the 'Plot' button is pressed.
  - ![Capture 1](https://user-images.githubusercontent.com/45400093/116785554-db4e2200-aab7-11eb-8e61-81da326aa65e.PNG)
- College Event Query:
  - Queries such as 'What are the events happening in harvard university?', 'Name some cycling events happening in US colleges', 'List the events happening in western region', etc. enter this flow.
  - Any question with keywords such as 'event', 'events', 'fair', 'competition', 'party', etc enter this flow.
  - The chatbot retrives information from the dataset and gives a reply.
  - The chatbot additionally provides a fun fact.(either related to the topic that is repeatedly asked or a randomly generated fun fact)
- College-career Query:
  - Queries such as 'What is the salary of a student graduating from brown university?', 'Can you tell me which is the best college in california?', 'What is the salary of a student graduating with a Art History major?', 'What is the average salary of a graduate in midwestern region?'. etc. enter this flow.
  - Any question with keywords that are not part of the above enter this flow.
  - The chatbot retrives information from the dataset and gives a reply.
  - The chatbot additionally provides a fun fact.(either related to the topic that is repeatedly asked or a randomly generated fun fact)
- Normal Chatbot Conversation:
  - Queries that do not have any specific keywords that are required to trigger the above flows enter this flow.
  - The chatbot is trained with chatterbot-corpus english and can give replies on various topics such as AI, science, philosophy, pscyhology, trivia, sports, etc.

## Future:
Currently, much of the code is hard coded with the dataset but in the future I would like to make an FAQ chatbot that can work with any dataset. This goal is partially implemented with the fun fact generation code. This code provides fun facts which are essentially values from the dataset which are either randomly taken from the dataset or taken after the user repeatedly asks for a certain type of query.

## Credits:
- Datasets:
  - [College-career dataset](https://www.kaggle.com/wsj/college-salaries)
  - [College Admission Prediction Dataset](https://www.kaggle.com/mohansacharya/graduate-admissions)
  - College Event dataset was a fake dataset created for this chatbot
- Chatterbot:
  - [Chatterbot Tutorial](https://chatterbot.readthedocs.io/en/stable/tutorial.html)
