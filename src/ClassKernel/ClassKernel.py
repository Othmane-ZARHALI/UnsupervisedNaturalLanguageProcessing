#Project : Unsupervised Natural Language Processing Project
#Date: 19/04/2021
#Author: Othmane ZARHALI
#Content: the class kernel file


#Package third parties
import json
import numpy as np


class DataPreprocessor():
    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def GetData(self):
        '''
            retrieves raw data from json
        '''
        f = open(self.file_path, encoding="utf8")
        data = json.load(f)
        X = []
        if 'texts' not in list(data.keys()):
            raise ValueError('Data Preprocessor error : texts is not in the json file')
        for item in data['texts']:
            if 'bbox' not in list(item.keys()):
                raise ValueError('Data Preprocessor error : bbox is not in the text item')
            position = item['bbox']
            X.append([position[0], position[1], position[2], position[3], item['nature'], item['text']])
        f.close()
        return X

    def Preprocess(self):
        '''
            retrieves raw data from json and then performs normalization on X & Y coordinates, as well as the text length
            we do not normalize the 'nature'.
        '''
        f = open(self.file_path)
        data = json.load(f)
        X = []
        TextSize = []
        if 'texts' not in list(data.keys()):
            raise ValueError('Data Preprocessor error : texts is not in the json file')
        for item in data['texts']:
            if 'bbox' not in list(item.keys()):
                raise ValueError('Data Preprocessor error : bbox is not in the text item')
            position = item['bbox']
            TextSize.append(len(item['text']))
            X.append([position[0] / 10000, -position[1] / 10000, position[2] / 10000, -position[3] / 10000, item['nature'],0])
        f.close()
        max_text_size = max(TextSize)
        X = np.array(X)
        TextSize = np.array(TextSize)
        X[:, 5] = TextSize / max_text_size
        return X

class Distance():
    def __init__(self,designation,available_designations=["X_distance","Y_distance","same side","text similarity","nature similarity"]):
        self.available_designations = available_designations
        if designation not in self.available_designations:
            raise ValueError('Distance error : designation not available')
        else:
            self.designation = designation

    def _distance(self,x1,x2,y1,y2):
        return max((abs(x1 + x2 - y1 - y2) - abs(x1 - x2) - abs(y1 - y2)) / 2.0, 0)

    def Construction(self,x,y):
        if self.designation == self.available_designations[0]:
            return self._distance(x[0],x[2],y[0],y[2])
        if self.designation == self.available_designations[1]:
            return self._distance(x[1],x[3],y[1],y[3])
        if self.designation == self.available_designations[2]:
            if (x[0] + x[2] < 1 and y[0] + y[2] < 1) or (x[0] + x[2] > 1 and y[0] + y[2] > 1):
                return True
            return False
        if self.designation == self.available_designations[3]:
            return min(x[5],y[5])/max(x[5],y[5])
        if self.designation == self.available_designations[4]:
            # case 1: x and y have the same nature
            if x[4] == y[4]:
                return True
            # case 2: x and y have the perfect Y alignement.
            # for instance, if paragraph is on the left side of the pdf and table on the right side
            # then x in the paragraph and y in the table will probably not have a perfect alignement
            elif x[1] == y[1] and x[3] == y[3]:
                return True
            else:
                return False
