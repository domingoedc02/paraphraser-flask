from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences

app = Flask(__name__)
api = Api(app)
CORS(app)
# Members API Route

# Paraphrasing download
model_name = 'tuner007/pegasus_paraphrase'

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(
    model_name).to(torch_device)


def get_response(input_text, num_return_sequences):
    batch = tokenizer.prepare_seq2seq_batch(
        [input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, max_length=60, num_beams=10,
                                num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(
        translated, skip_special_tokens=True)
    return tgt_text


Data = []


class Paragraph(Resource):
    def post(self):
        data = request.json
        temp = {'Text': data['Text']}
        Data.append(temp)

        # summarizing code
        stopwords = list(STOP_WORDS)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(temp['Text'])
        tokens = [token.text for token in doc]
        print(tokens)
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in stopwords:
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1

        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word]/max_frequency
        print(word_frequencies)
        sentence_tokens = [sent for sent in doc.sents]
        sentence_scores = {}

        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        select_length = int(len(sentence_tokens)*0.3)
        summary = nlargest(select_length, sentence_scores,
                           key=sentence_scores.get)
        final_summary = [word.text for word in summary]
        summary = ' '.join(final_summary)

        return summary


class Paraphrase(Resource):
    def post(self):
        data = request.json
        temp = {'Text': data['Text']}
        Data.append(temp)
        context = temp['Text']

        splitter = SentenceSplitter(language='en')
        sentence_list = splitter.split(context)
        sentence_list

        paraphrase = []
        for i in sentence_list:
            a = get_response(i, 1)
            paraphrase.append(a)
        paraphrase2 = [' '.join(x) for x in paraphrase]
        paraphrase3 = [' '.join(x for x in paraphrase2)]
        paraphrased_text = str(paraphrase3).strip('[]').strip("'")
        final = paraphrased_text.strip('\"')

        return final


api.add_resource(Paragraph, '/summarize')
api.add_resource(Paraphrase, '/paraphrase')
if __name__ == "__main__":
    app.run(debug=True)
