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
- sklearn Module
  - `pip install sklearn`
- Download all the files in the folder and keep them in the same workspace

## Working:
- Run the `Higher_Education_Chatbot.py` file on your python IDE
- A tkinter GUI with a chatbot type interface and a plot button on the right should appear

## Credits:
- Datasets:
  - [College-career dataset](https://www.kaggle.com/wsj/college-salaries)
  - [College Admission Prediction Dataset](https://www.kaggle.com/mohansacharya/graduate-admissions)
  - College Event dataset was a fake dataset created for this chatbot
- Chatterbot:
  - [Chatterbot Tutorial](https://chatterbot.readthedocs.io/en/stable/tutorial.html)
