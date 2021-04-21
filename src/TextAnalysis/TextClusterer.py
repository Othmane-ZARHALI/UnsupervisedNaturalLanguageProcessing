#Project : Unsupervised Natural Language Processing Project
#Date: 19/04/2021
#Author: Othmane ZARHALI
#Content: the text clustering algorithm


#Package third parties
from ClassKernel import *
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class Metric():
    def __init__(self,designation,coefficients={'coef_x':[0.1 ,0.5],'coef_y':[1.5,2,2.5],'nature_penalty':[-0.01,0.02],'similarity_threashold':[0.2,0.4]},available_designations=["nature","OurMetric","MultiplicativeOfOurMetric"]):
        if designation not in available_designations:
            raise ValueError('Metric error : designation not in available designations - "nature" and "Lunalogicmetric"')
        else:
            self.designation = designation
        self.available_designations = available_designations
        if type(coefficients) != list and type(coefficients) != dict:
            raise ValueError('Metric error : coefficients not of expected type (list or dict)')
        else:
            self.coefficients = coefficients #{'coef_x':[0.1 ,0.5],'coef_y':[1.5,2,2.5],'nature_penalty':[-0.01,0.02],'similarity_threashold':[0.2,0.4]}
    def Construction(self,x, y):
        if self.designation == self.available_designations[0]:
            '''
                   basic metric to illustrate clustering by 'nature'
            '''
            if x[4] == y[4]:
                return 0
            else:
                return 1
        if self.designation in  {self.available_designations[1],self.available_designations[2]}:
            '''
                   custom DBSCAN metric 
            '''
            X_distance,Y_distance,nature_similarity_distance,text_similarity_distance,same_side_distance = Distance("X_distance"),Distance("Y_distance"),Distance("nature similarity"),Distance("text similarity"),Distance("same side")
            ### calculate basic distances ###
            dist_x = X_distance.Construction(x, y)
            dist_y = Y_distance.Construction(x, y)
            ### determine coef_x ###
            if nature_similarity_distance.Construction(x, y):
                coef_x = self.coefficients['coef_x'][0]
                nature_penalty = self.coefficients['nature_penalty'][0]
            else:
                coef_x = self.coefficients['coef_x'][1]
                nature_penalty = self.coefficients['nature_penalty'][1]
            # x and y are on the same side of the pdf (left and right)
            if same_side_distance.Construction(x, y):
                coef_x = 1
            ### determine coef_y ###
            coef_y = 1
            similarity = text_similarity_distance.Construction(x, y)
            if similarity < self.coefficients['similarity_threashold'][0]:
                coef_y = self.coefficients['coef_y'][0]
            elif similarity < self.coefficients['similarity_threashold'][1]:
                coef_y = self.coefficients['coef_y'][1] - self.coefficients['coef_y'][2] * similarity
            if self.designation == self.available_designations[1]:
                return coef_x * dist_x + coef_y * dist_y + nature_penalty
            if self.designation == self.available_designations[2]:
                return (coef_x * dist_x + coef_y * dist_y )* nature_penalty

class ClusteringHyperParameterTuner():
    def __init__(self,hyper_parameters_ranges):
        if type(hyper_parameters_ranges) != dict:
            raise ValueError('Hyper Parameter Tuner error : hyper_parameters_ranges not of expected type - dict')
        self.hyper_parameters_ranges = hyper_parameters_ranges

    def HPClustersComparator(self, Json1, Json2):
        """returns the number of common clusters"""
        clusters1,clusters2 = [],[]
        for item1,item2 in zip(Json1['texts'],Json2['texts']):
            clusters1.append(item1['Cluster label']),clusters2.append(item2['Cluster label'])
        return len(list(set(clusters1).intersection(clusters2)))

    def Perform(self, perfect_json, raw_json, data, eps, min_samples, metric_type='internal', useOriginalLabels=False,
                graph=True, parameter_tunning_time_performance=True):
        if metric_type != 'internal':
            raise ValueError('Hyper Parameter Tuner error : only internal metrics are hyper tuned')
        if len(list(self.hyper_parameters_ranges.values())) != 9:
            raise ValueError('Hyper Parameter Tuner error : ranges are not of expected length')
        else:
            # length hyper parameter range checking:
            initial_length = len(list(self.hyper_parameters_ranges.values())[0])
            for x in list(self.hyper_parameters_ranges.values()):
                if len(x) != initial_length:
                    raise ValueError('Hyper Parameter Tuner error : ranges must have equal lengths')
            # target search ranges constitutions:
            target_values = []
            if parameter_tunning_time_performance == False:
                target_search_list = list(zip(*list(self.hyper_parameters_ranges.values())))
                for items in target_search_list:
                    metric = Metric("OurMetric", {'coef_x': [items[0], items[1]],
                                                        'coef_y': [items[2], items[3], items[4]],
                                                        'nature_penalty': [items[5], items[6]],
                                                        'similarity_threashold': [items[7], items[8]]})
                    model = Clusterer(raw_json, data, eps, min_samples, metric, metric_type, useOriginalLabels, graph)
                    output_json = model.OuputPostprocessingClusteringResults(False)
                    target_values.append(self.HPClustersComparator(perfect_json, output_json))
                return target_search_list[target_values.index(max(target_values))]


            else:
                time_now = datetime.datetime.now().timestamp()
                target_search_list = list(zip(*list(self.hyper_parameters_ranges.values())))
                for items in target_search_list:
                    metric = Metric("OurMetric", {'coef_x': [items[0], items[1]],
                                                        'coef_y': [items[2], items[3], items[4]],
                                                        'nature_penalty': [items[5], items[6]],
                                                        'similarity_threashold': [items[7], items[8]]})
                    model = Clusterer(raw_json, data, eps, min_samples, metric, metric_type, useOriginalLabels, graph)
                    output_json = model.OuputPostprocessingClusteringResults(False)
                    target_values.append(self.HPClustersComparator(perfect_json, output_json))
                return target_search_list[target_values.index(max(target_values))], (
                            datetime.datetime.now().timestamp() - time_now)


class Clusterer():
    def __init__(self,raw_json,data,eps, min_samples, metric, metric_type = 'internal', useOriginalLabels=False, graph=False):
        self.raw_json = raw_json #raw_json is of the form json.load(path file)
        self.data = data
        self.radius_min = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_type = metric_type # 'internal' or 'external'
        self.useOriginalLabels = useOriginalLabels
        self.graph = graph

    def Create(self):
        if self.metric_type == 'internal':
            if self.metric.designation not in self.metric.available_designations:
                raise ValueError(
                    'Clusterer error : the Clusterer metric type is not inline with the available metrics - define your external metric')
            else:
                dbscan = DBSCAN(eps=self.radius_min, min_samples=self.min_samples, metric=self.metric.Construction,
                                algorithm='brute')  # self.metric is a instance of Metric
        else:
            dbscan = DBSCAN(eps=self.radius_min, min_samples=self.min_samples, metric=self.metric,
                            algorithm='brute')  # self.metric is a Method
        return dbscan

    def Perform(self,clustering_time_performance_flag=True):
        '''
               - perform DBSCAN algorithm
               - improve DBSCAN results with original clusters
               - output results & plot if required
        '''
        output_dict = dict()
        if clustering_time_performance_flag == True:
            time_now = datetime.datetime.now().timestamp()
            # core start : pre-processing + DBSCAN + redistribution of labels
            # X = preprocess(file)
            dbscan = self.Create()
            dbscan.fit(self.data)
            labels = dbscan.labels_
            if self.useOriginalLabels:
                unique_labels = set(self.data[:, 4])
                current_labels = set(labels)
                new_labels = np.zeros_like(labels, dtype=int)
                for k in current_labels:
                    class_member_mask = (labels == k)
                    current_members = self.data[class_member_mask]
                    original_labels, counts = np.unique(current_members[:, 4], return_counts=True)
                    max_idx = np.where(counts == np.amax(counts))
                    new_labels[class_member_mask] = original_labels[max_idx[0][0]]
            else:
                unique_labels = set(labels)
                new_labels = labels
            # core end : DBSCAN + redistribution of labels
            output_dict["Clustering time spent"] = datetime.datetime.now().timestamp() - time_now
            # output
            n_clusters_ = len(set(new_labels)) - (1 if -1 in new_labels else 0)
            n_noise_ = list(new_labels).count(-1)
            output_dict["Sample size"] = len(self.data)
            output_dict["Estimated number of clusters"] = n_clusters_
            output_dict["Estimated number of noise points"] = n_noise_
        else:
            # core start : pre-processing + DBSCAN + redistribution of labels
            # X = preprocess(file)
            dbscan = self.Create()
            dbscan.fit(self.data)
            labels = dbscan.labels_
            if self.useOriginalLabels:
                unique_labels = set(self.data[:, 4])
                current_labels = set(labels)
                new_labels = np.zeros_like(labels, dtype=int)
                for k in current_labels:
                    class_member_mask = (labels == k)
                    current_members = self.data[class_member_mask]
                    original_labels, counts = np.unique(current_members[:, 4], return_counts=True)
                    max_idx = np.where(counts == np.amax(counts))
                    new_labels[class_member_mask] = original_labels[max_idx[0][0]]
            else:
                unique_labels = set(labels)
                new_labels = labels
            # core end : DBSCAN + redistribution of labels
            # output
            n_clusters_ = len(set(new_labels)) - (1 if -1 in new_labels else 0)
            n_noise_ = list(new_labels).count(-1)
            output_dict["Sample size"] = len(self.data)
            output_dict["Estimated number of clusters"] = n_clusters_
            output_dict["Estimated number of noise points"] = n_noise_
        if self.graph:
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[dbscan.core_sample_indices_] = True
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                class_member_mask = (new_labels == k)
                xy = self.data[class_member_mask & core_samples_mask]
                plt.plot((xy[:, 0] + xy[:, 2]) / 2.0, (xy[:, 1] + xy[:, 3]) / 2.0, 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=14)
                xy = self.data[class_member_mask & ~core_samples_mask]
                plt.plot((xy[:, 0] + xy[:, 2]) / 2.0, (xy[:, 1] + xy[:, 3]) / 2.0, 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.show()
        return self.data, new_labels, output_dict

    def OuputPostprocessingClusteringResults(self,overall_time_performance = True):
        X = self.raw_json
        if overall_time_performance == True:
            new_labels, Clustering_time = self.Perform(True)[1],self.Perform(True)[2]["Clustering time spent"]
            time_now = datetime.datetime.now().timestamp()
            for item in X['texts']:
                item['Cluster label'] = new_labels[list(X.keys()).index(item)]
            return X,Clustering_time+(datetime.datetime.now().timestamp()-time_now)
        else:
            new_labels = self.Perform(True)[1]
            for item in X['texts']:
                item['Cluster label'] = int(new_labels[X['texts'].index(item)])
            return X




