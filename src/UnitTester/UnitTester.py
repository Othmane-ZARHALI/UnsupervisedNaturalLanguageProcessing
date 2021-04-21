#Project : Unsupervised Natural Language Processing Project
#Date: 19/04/2021
#Author: Othmane ZARHALI
#Content: the unit testing file


#Package third parties
import copy
import csv
from TextClusterer import *

class Test():
    def __init__(self, list_jsons, eps, min_samples, metric, metric_type, overall_time_performance, useOriginalLabels=False):
        self.list_jsons = list_jsons[:]
        self.eps = eps
        self.min_samples = min_samples
        self.metric = copy.deepcopy(metric)
        self.metric_type = metric_type
        self.overall_time_performance = overall_time_performance
        self.useOriginalLabels = useOriginalLabels

    def generateJsons(self):
        list_output_jsons = []
        for file_path in self.list_jsons:
            dataPreprocessor = DataPreprocessor(file_path)
            f = open(file_path)
            raw_json = json.load(f)
            data = dataPreprocessor.Preprocess()
            dbscan = Clusterer(raw_json, data, self.eps, self.min_samples, self.metric, self.metric_type, self.useOriginalLabels)
            if self.overall_time_performance:
                new_json, executionTime = dbscan.OuputPostprocessingClusteringResults(True)
            else:
                new_json = dbscan.OuputPostprocessingClusteringResults(False)

            new_file_path = file_path[:-5] + "_new.json"
            json_object = json.dumps(new_json, indent=4)
            list_output_jsons.append(json_object)
            with open(new_file_path, "w") as outfile:
                outfile.write(json_object)
        return list_output_jsons



class Regression():
    def __init__(self, list_json_paths, list_json_benchmark_paths, list_parametrization1, list_parametrization2):
        self.list_json_paths = list_json_paths[:]
        self.list_json_benchmark_paths = list_json_benchmark_paths[:]
        self.list_parametrization1 = list_parametrization1[:]
        self.list_parametrization2 = list_parametrization2[:]

    def _getClustersFile(self, file_path):
        """returns clusters of a json output"""
        data = json.load(open(file_path))
        clusters = []
        for item in data['texts']:
            clusters.append(item['Cluster label'])
        return clusters

    def _getClustersJson(self, json_file):
        """returns clusters of a json output"""
        clusters = []
        loaded_json = json.loads(json_file)
        for item in loaded_json['texts']:
            clusters.append(item['Cluster label'])
        return clusters

    def _compareClusters(self, clusters1, clusters2, parametrization1, parametrization2):
        """compares clusters of 2 json output files and returns list of clusters occurences"""
        setClusters1 = set(clusters1)
        setClusters2 = set(clusters2)
        output = [['Comparison results betwen %s and %s' %(parametrization1, parametrization2)]]
        if setClusters1 != setClusters2:
            output.append(['The sets of clusters are not the same so the json files cannot be compared'])
            return output
        output.append(['Cluster number', 'Test Json', 'Benchmarked Json'])
        for cluster in setClusters1:
            output.append([cluster, clusters1.count(cluster), clusters2.count(cluster)])
        return output

    def calculateRegression(self):
        for i in range(len(self.list_json_paths)):
            clusters1 = self._getClustersJson(self.list_json_paths[i])
            clusters2 = self._getClustersFile(self.list_json_benchmark_paths[i])
            parametrization1 = self.list_parametrization1[i]
            parametrization2 = self.list_parametrization2[i]
            final_output = self._compareClusters(clusters1, clusters2, parametrization1, parametrization2)
            with open('results.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(final_output)





