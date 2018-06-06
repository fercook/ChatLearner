# Copyright 2017 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import print_function
import os
import sys
import tensorflow as tf

from settings import PROJECT_ROOT
from chatbot.botpredictor import BotPredictor
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def bot_ui():
    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')

    vocab = []
    with open(os.path.join(corp_dir,"vocab.txt")) as fi:
        for line in fi:
            vocab.append(line.strip())
        
    with tf.Session() as sess:
        predictor = BotPredictor(sess, corpus_dir=corp_dir, knbase_dir=knbs_dir,
                                 result_dir=res_dir, result_file='basic')
        # This command UI has a single chat session only
        session_id = predictor.session_data.add_session()

        print("Welcome to Chat with ChatLearner!")
        print("Type exit and press enter to end the conversation.")
        # Waiting from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        question = sys.stdin.readline()
        while question:
            if question.strip() == 'exit':
                print("Thank you for using ChatLearner. Goodbye.")
                break

            answer, other_choices, probabilities = predictor.predict(session_id, question, fullResponse=True)
            print (answer)
            print ([ {"word": vocab[other_choices[n][0][0]], "prob": "{0:.4f}".format(math.exp(probabilities[n][0][0]))} for n in range(len(other_choices)) if vocab[other_choices[n][0][0]]!= '_eos_'  ])
#            for depth in range(len(other_choices)):
#                print ( [ (vocab[word],math.exp(prob)) for word,prob in zip( other_choices[depth][0], probabilities[depth][0] ) ] )
            print("> ", end="")
            sys.stdout.flush()
            question = sys.stdin.readline()

if __name__ == "__main__":
    bot_ui()
