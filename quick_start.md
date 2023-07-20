## Requirements

- Python 3.8+
- Rasa 2.8.2

## Installation

1. Install the required Python packages:
   
   ```
   pip install -r requirements.txt
   ```
2. Put the openai.key to the credentials folder
 
   ```
   chatbot-nlu-enhancement/
     ...
     ├── credentials/openai.key
     ├── README.md
     ├── requirements.txt
     ...
   ```

## Run Client and Server

1. Run the Streamlit app:
   
   ```python
   streamlit run app.py
   ```

2. Open another two terminals, and go to the `rasa_server` folder
   ```
   cd rasa_server
   ```

3. Run these two commands in different terminals
   ```
   python run.py --port=5623
   ```
   ```
   rasa run actions --port=5624
   ```

## Usage

1. Open the app in your browser:   
   http://localhost:8501

2. (Optional)Put some article text files under the `articles` folder. <br>
   Or write the article on the app page.

3. Press the button `Generate NLU !` <br>
   Note: if it shows parsing error, please press the button again.

4. Press the button `Train NLU and Activate Model` <br>

5. Once the model is trained and activated, open the [webchat](./rasa_server/index.html) in the browser

6. Talk to Chat bot 🤖
