from oauth2client.service_account import ServiceAccountCredentials
from datetime import date
from model import model
import pandas as pd
import numpy as np
import datetime
import gspread
import logging
import pickle
import json
import os


class SlackMessageHelper:
    def __init__(self):
        self.loaded_model = pickle.load(open('./model/model.pkl', 'rb'))
        self.encoded_intents = pd.read_csv('./data/encoded_labels.csv')
        self.vectorizer = pickle.load(open('./model/tfidf-vectorizer.pkl', 'rb'))
        self.count_vectorizer = pickle.load(open('./model/count-vectorizer.pkl', 'rb'))
        with open('./data/best_model.txt') as f:
            self.best_model = f.read()
        with open('channels_info.json') as f:
            self.channels_info = json.load(f)


    def get_response(self, message):
        """Returns appropriate response for the given message"""

        intent, prob = authorization_helper._analyze_message(message)
        logging.info(f"Message: {message} - Prediction: {intent} - Probability: {prob}")

        if DEBUG:
            return f"Prediction: {intent}", intent, prob

        if intent == 'authorization':
            # TODO Take action here
            return 'Response message for authorization requests.', intent, prob
        elif intent == 'catalog':
            # TODO Take action here
            return 'Response message for catalog questions.', intent, prob
        elif intent == 'irrelevant':
            # TODO Take action here
            return '', intent, prob
        elif  intent == 'troubleshooting':
            # TODO Take action here
            return 'Response message for troubleshooting.', intent, prob
        elif  intent == 'no-reply':
            # TODO Take action here
            return '', intent, prob

    def _analyze_message(self, message):
        """Predict the message's intent by using model"""

        message = model.preprocessing(message)

        if 'tfidf' in self.best_model:
            tfidf_matrix = self.vectorizer.transform([message])
            prediction = self.loaded_model.predict(tfidf_matrix)
            prediction_probability = np.max(self.loaded_model.predict_proba(tfidf_matrix))
        elif 'count' in self.best_model:
            count_vector = self.count_vectorizer.transform([message])
            prediction = self.loaded_model.predict(count_vector)
            prediction_probability = np.max(self.loaded_model.predict_proba(count_vector))

        prediction = self.encoded_intents[self.encoded_intents['label_enc'] == prediction[0]]['label'].iloc[0]

        return prediction, prediction_probability

    def get_sheet_cover(self, text, intent, channel, user_name, ts, prob):
        """Connect the google sheet and save information"""

        scope = ["https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CLIENT_SECRET_FILE, scope)
        gs_client = gspread.authorize(creds)

        sheet = gs_client.open("babur-bot-messages").get_worksheet(0)

        updating_message_cell = 'C' + str(len(sheet.col_values(3))+1) ## message column
        sheet.update(updating_message_cell, text)

        updating_prediction_cell = 'E'+str(len(sheet.col_values(3))) ## prediction column
        sheet.update(updating_prediction_cell, intent)

        updating_ts_cell = 'F'+str(len(sheet.col_values(3))) ## ts column
        sheet.update(updating_ts_cell, ts)

        updating_time_cell = 'G'+str(len(sheet.col_values(3))) ## time column
        sheet.update(updating_time_cell, datetime.datetime.now().isoformat())

        updating_channel_cell = 'A'+str(len(sheet.col_values(3))) ## channel column
        sheet.update(updating_channel_cell, self.channels_info[channel])

        updating_name_cell = 'B'+str(len(sheet.col_values(3))) ## name column
        sheet.update(updating_name_cell, user_name)

        updating_prob_cell = 'H'+str(len(sheet.col_values(3))) ## probability column
        sheet.update(updating_prob_cell, str(prob))

        updating_link_cell = 'I'+str(len(sheet.col_values(3))) ## slack message link
        ts = str(ts).replace('.','')
        sheet.update(updating_link_cell, f'https://dummy-domain.slack.com/archives/{channel}/p{ts}')
