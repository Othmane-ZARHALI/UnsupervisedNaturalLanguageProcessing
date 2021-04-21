#Project : Unsupervised Natural Language Processing Project
#Date: 19/04/2021
#Author: Othmane ZARHALI
#Content: the text processing post clustering


#Package third parties
import json


class TextRegrouper():
    def __init__(self,original_data_file_path,json_output):
        self.original_data_file_path = original_data_file_path
        json_output_loaded = json.loads(json_output)
        if json_output_loaded['texts'] == []:
            raise ValueError('Text Regrouper error : json output is not of expected content')
        else:
            if 'Cluster label' not in list(json_output_loaded['texts'][0].keys()):
                raise ValueError('Text Regrouper error : Cluster label must be in the json texts')
        self.json_output = json_output

    def Jsonloader(self):
        return json.loads(self.json_output)

    def LocalgetClustersJson(self):
        """returns clusters of a json output"""
        clusters = []
        loaded_json = self.Jsonloader()
        for item in loaded_json['texts']:
            clusters.append(item['Cluster label'])
        return clusters

    def Perform(self):
        labels = set(self.LocalgetClustersJson())
        regroupped_texts = dict()
        for label in labels:
            regroupped_texts[label] = str()
        loaded_json = self.Jsonloader()
        for item in loaded_json['texts']:
            if item['Cluster label'] not in labels:
                raise ValueError('Text Regrouper error : the label is not in the set of clusters')
            else:
                regroupped_texts[item['Cluster label']] += item['text']
        return regroupped_texts

    def RegroupWords(self, text):
        words = []
        word = str()
        for x in text:
            if x != ' ':
                word += x
            else:
                words.append(word)
                word = str()
        return words

    def OccurenceTextSynthesis(self,text,threashold_occurence,word_length):
        dict_target = dict()
        words = self.RegroupWords(text)
        for x in words:
            if len(x)>=word_length:
                dict_target[x] = words.count(x)
        return dict((k,v)for k, v in dict_target.items() if v >= threashold_occurence)

    def Processor(self,threashold_occurence,word_length):
        regroupped_texts = self.Perform()
        synthesis_texts = dict()
        for key in list(regroupped_texts.keys()):
            synthesis_texts[key] = self.OccurenceTextSynthesis(regroupped_texts[key],threashold_occurence,word_length)
        return dict((k, v) for k, v in synthesis_texts.items() if list(v.values()) != [])





