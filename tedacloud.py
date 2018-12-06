import numpy as np
import matplotlib.pyplot as plt
import itertools
from pprint import pprint
from sklearn.datasets import load_iris
import pandas as pd
import csv
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn import metrics
import copy
import tqdm
import time
from os.path import join


def gen_Label_Solution(centers, labels):
    # generate unique label solution (mapping for comparison between different label solutions)
    idx = np.argsort(centers.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(len(centers))
    return(lut[labels])

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if 'feature' in feature_name:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

class Tedacloud:
    # teda-cloud class
    
    def __init__(self, dataset):
        # load dataset and get feature list
        self.dataset = dataset
        self.features = np.sort([feature for feature in self.dataset.columns if 'feature' in feature])

    def recursive_teda(self, curr_observation, previous_mean, previous_var = 0, previous_scal = 0, k = 1, dist_type = 'euclidean', m=3):
        
        recursive_mean = ((k - 1) / k)*previous_mean + (1/k)*curr_observation
        
        if k == 1:
            recursive_var = previous_var
        else:
            recursive_var = ((k - 1)/k)*previous_var + (1/(k-1)) * np.sum(np.power(curr_observation-recursive_mean,2))
        
        recursive_ecc = (1/k) + (np.sum((recursive_mean - curr_observation) * (recursive_mean - curr_observation))/(k * recursive_var))

        ret_obj = {}
        ret_obj['curr_observation'] = curr_observation
        ret_obj['curr_mean'] = recursive_mean
        ret_obj['curr_var'] = recursive_var
        
        #ret_obj['curr_scal'] = recursive_scal
        ret_obj['curr_scal'] = 0

        ret_obj['curr_eccentricity'] = recursive_ecc
        #ret_obj['curr_typicality'] = 1 - recursive_ecc
        ret_obj['curr_norm_eccentricity'] = ret_obj['curr_eccentricity']/2
        #ret_obj['curr_norm_typicality'] = ret_obj['curr_typicality']/(k - 2)
        ret_obj['outlier'] = ret_obj['curr_norm_eccentricity'] > (m*m+1)/(2*k)
        #ret_obj['ecc_threshold'] = 1/k
        ret_obj['next_k'] = k + 1
        return(ret_obj)

    def assign_points_to_unique_clouds(self):
        # assign points to unique clouds (based on proximity to closest center) in order to compare two clustering solution
        
        # make a deep copy of clus_teda and dic_label_solution in order to create objects with unique points assigned to clouds
        self.dic_new_label_solution = copy.deepcopy(self.dic_label_solution)
        self.new_clus_teda = copy.deepcopy(self.clus_teda)

        for comb in list(itertools.combinations(self.dic_new_label_solution.keys(), 2)):
            s1 = set(self.dic_new_label_solution[comb[0]])
            s2 = set(self.dic_new_label_solution[comb[1]])
            list_intersec = s1.intersection(s2)

            for intersec in list_intersec:
                #print('Interseção de pontos detectada entre cloud {0} e cloud {1}'.format(comb[0],comb[1]))
                d1 = np.sqrt(np.sum(self.dataset.loc[intersec,self.features] * self.new_clus_teda[comb[0]]['curr_mean']))
                d2 = np.sqrt(np.sum(self.dataset.loc[intersec,self.features] * self.new_clus_teda[comb[1]]['curr_mean']))

                if d1 < d2:
                    #print('Interseção de pontos detectada entre cloud {0} e cloud {1}'.format(comb[0],comb[1]))
                    self.dic_new_label_solution[comb[1]].remove(intersec)
                    # falta atualizar os parametros
                    self.new_clus_teda[comb[1]]['next_k'] -= 1
                else:
                    #print('Interseção de pontos detectada entre cloud {0} e cloud {1}'.format(comb[0],comb[1]))
                    self.dic_new_label_solution[comb[0]].remove(intersec)
                    # falta atualizar os parametros
                    self.new_clus_teda[comb[0]]['next_k'] -= 1

    def adjust_label_solution(self):
        # generate unique label solution (mapping for comparison between different label solution)
        list_new_keys = np.arange(1,len(self.new_clus_teda.keys())+1,1)
        list_old_keys = np.sort(list(self.new_clus_teda.keys()))
        for old, new in zip(list_old_keys,list_new_keys):
            if old != new:
                self.new_clus_teda[new] = self.new_clus_teda.pop(old)
                self.dic_new_label_solution[new] = self.dic_new_label_solution.pop(old)
        
        self.clouds_centers = []
        for key in self.new_clus_teda.keys():
            self.clouds_centers.append(self.new_clus_teda[key]['curr_mean'].values)
        self.clouds_centers = np.sort(self.clouds_centers, axis=0)

        self.label_solution = []
        for obj in self.dataset.index.values:
            for key in self.new_clus_teda.keys():
                if obj in self.dic_new_label_solution[key]:
                    self.label_solution.append(key)

        self.label_solution_adjusted = self.label_solution
        
        idx = np.argsort(self.clouds_centers.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(len(self.clouds_centers))
        self.label_solution = np.array(self.label_solution) - 1
        self.label_solution_adjusted = lut[self.label_solution]
        self.label_solution_adjusted += 1
        
        
    def Fit(self, sd):

        # Definition of tedacloud and object dics
        self.clus_teda = {}
        self.dic_label_solution = {}
        self.nclust = 0
        self.nmerge = 0
        
        nobj = len(self.dataset['Label'].values)
        for k in tqdm.tqdm_notebook(range(1,nobj+1)):
            # Case 1: k equals to 1
            if(k == 1):
                self.nclust = 1
                self.dic_label_solution[1] = [k]
                self.clus_teda[1] = self.recursive_teda(self.dataset.loc[k,self.features],
                                                        previous_mean = self.dataset.loc[k,self.features], 
                                                        m = sd)

            # Case 2: k equals to 2
            elif(k == 2):
                self.nclust = 1
                self.dic_label_solution[1].append(k)
                self.clus_teda[1] = self.recursive_teda(self.dataset.loc[k,self.features],
                                              self.clus_teda[1]['curr_mean'], 
                                              self.clus_teda[1]['curr_var'],
                                              self.clus_teda[1]['curr_scal'],
                                              self.clus_teda[1]['next_k'],
                                              m = sd)
            # Case 3: k greater than 3
            else:
                # Verify if the point belongs to any cloud.
                count = 0
                for key in list(self.clus_teda.keys()):
                    aux_teda = self.recursive_teda(self.dataset.loc[k,self.features],
                                              self.clus_teda[key]['curr_mean'],
                                              self.clus_teda[key]['curr_var'],
                                              self.clus_teda[key]['curr_scal'],
                                              self.clus_teda[key]['next_k'],
                                              m = sd)

                    if(aux_teda['outlier'] == 0):
                        self.clus_teda[key] = aux_teda
                        self.dic_label_solution[key].append(k)
                        count += 1
                
                # If the point does not belong to any cloud, a new one cloud is created.
                if(count == 0):
                    self.nclust += 1
                    if self.nclust not in self.clus_teda.keys():
                        self.clus_teda[self.nclust] = self.recursive_teda(self.dataset.loc[k,self.features], previous_mean = self.dataset.loc[k,self.features], m = sd)
                        self.dic_label_solution[self.nclust] = [k]
                    else:
                        new_nclust = max(self.clus_teda.keys())+1
                        self.clus_teda[new_nclust] = self.recursive_teda(self.dataset.loc[k,self.features], previous_mean = self.dataset.loc[k,self.features], m = sd)
                        self.dic_label_solution[new_nclust] = [k]

            # Check if there are clouds to merge
            comb = list(itertools.combinations(self.dic_label_solution.keys(), 2))
            idx = []
            for h in comb:
                if((len(self.clus_teda[h[0]])) > 1 and len(self.clus_teda[h[1]]) > 1):
                    s1 = set(self.dic_label_solution[h[0]])
                    s2 = set(self.dic_label_solution[h[1]])
                    nintersec = len(s1.intersection(s2))

                    n1 = len(s1) - nintersec
                    n2 = len(s2) - nintersec

                    # Test merge condition
                    if(nintersec > n1 or nintersec > n2):
                        self.nmerge += 1
                        self.nclust -= 1
                        self.dic_label_solution[h[0]] = list(s1.union(s2))

                        # Update new parameters
                        self.clus_teda[h[0]]['curr_observation'] = self.dataset.loc[k,self.features]
                        self.clus_teda[h[0]]['curr_mean'] = (n1*self.clus_teda[h[0]]['curr_mean'] + n2*self.clus_teda[h[1]]['curr_mean'])/(n1+n2)
                        self.clus_teda[h[0]]['curr_var'] = ((n1-1)*self.clus_teda[h[0]]['curr_var'] + (n2-1)*self.clus_teda[h[1]]['curr_var'])/(n1 + n2 - 2)
                        #self.clus_teda[h[0]]['next_k'] = len(self.dic_label_solution[h[0]]) + len(self.dic_label_solution[h[1]]) - nintersec
                        self.clus_teda[h[0]]['next_k'] = len(self.dic_label_solution[h[0]]) + len(self.dic_label_solution[h[1]]) - nintersec + 1

                        # Mark the second cloud for deletion process
                        self.clus_teda[h[1]] = []
                        idx.append(h[1])
            
            # Delete clouds after merging process
            for dlt in idx:
                del self.clus_teda[dlt]
                del self.dic_label_solution[dlt]
    
        # assign points to unique clouds
        self.assign_points_to_unique_clouds()
        
        # compute adjusted label solution
        self.adjust_label_solution()

def create_dataset(name_dataset):
    
    # dataset directory
    dir_datasets = 'datasets'

    if(name_dataset == 'gaussian'):
        # define means
        mean_1, sigma_1 = 0.5, 0.2
        mean_2, sigma_2 = 2.0, 0.4
        mean_3, sigma_3 = [3.0,3.0], 0.1

        # define points
        x11 = np.random.normal(mean_1, sigma_1, 1000)
        x12 = np.random.normal(mean_1, sigma_1, 1000)
        x21 = np.random.normal(mean_2, sigma_2, 1000)
        x22 = np.random.normal(mean_2, sigma_2, 1000)
        x31 = np.random.normal(mean_3[0], sigma_3, 1000)
        x32 = np.random.normal(mean_3[1], sigma_3, 1000)
        aux_1 = np.reshape(np.concatenate((x11,x12)),(len(x11),2))
        aux_2 = np.reshape(np.concatenate((x21,x22)),(len(x21),2))
        aux_3 = np.reshape(np.concatenate((x31,x32)),(len(x31),2))

        # define pandas dataframe
        df = pd.DataFrame(np.concatenate((aux_1,aux_2,aux_3)), columns=['feature_1','feature_2'])
        y = np.concatenate((np.ones(1000), 2*np.ones(1000), 3*np.ones(1000)))
        df['Label'] = y
        df['Label'] = df['Label'].astype(int)
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)

    elif(name_dataset == 'iris'):
        iris = load_iris()
        df = pd.DataFrame(iris.data[:, :], columns=['feature_1','feature_2','feature_3','feature_4'])
        df['Label'] = iris.target + 1
        df['Label'] = df['Label'].astype(int)
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)

    elif(name_dataset == 'smile'):
        df = pd.read_csv(join(dir_datasets,'smile.csv'))
        df.rename(columns={'X': 'feature_1', 'Y': 'feature_2'}, inplace=True)
        df['Label'] = df['Label'].astype(int)        
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)

    elif(name_dataset == 'spirals'):
        df = pd.read_csv(join(dir_datasets,'spirals.csv'))
        df.rename(columns={'X': 'feature_1', 'Y': 'feature_2'}, inplace=True)
        df['Label'] = df['Label'].astype(int)
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)

    elif(name_dataset == 'shapes'):
        df = pd.read_csv(join(dir_datasets,'shapes.csv'))
        df.rename(columns={'X': 'feature_1', 'Y': 'feature_2'}, inplace=True)
        df['Label'] = df['Label'].astype(int)
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)
    
    elif(name_dataset == 'a1'):
        df = pd.read_csv(join(dir_datasets,'a1.csv'), sep=';')
        df['Label'] = 1
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)

    elif(name_dataset == 'a2'):
        df = pd.read_csv(join(dir_datasets,'a2.csv'), sep=';')
        df['Label'] = 1
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)
        
    elif(name_dataset == 's1'):
        df = pd.read_csv(join(dir_datasets,'s1.csv'), sep=';')
        df['Label'] = 1

        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)

        df['feature_1'].astype(float)
        df['feature_1'] = df['feature_1']/np.max(df['feature_1'].values)
        df['feature_2'] = df['feature_2']/np.max(df['feature_2'].values)
        
    elif(name_dataset == 's2'):
        df = pd.read_csv(join(dir_datasets,'s2.csv'), sep=';')
        df['Label'] = 1
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)
        df['feature_1'].astype(float)
        df['feature_1'] = df['feature_1']/np.max(df['feature_1'].values)
        df['feature_2'] = df['feature_2']/np.max(df['feature_2'].values)
        
    elif(name_dataset == 'adl'):
        df = pd.read_csv(join(dir_datasets,'AccelerometerData_getup-bed_brush-teeth_eat-meat_f1.csv'), sep=';', index_col=None)
        df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
        df['index'] = df['index'].astype(int)
        df.set_index('index', inplace=True)

        
    return(df)

def sliding_window(df,window_size):
    
    idx = pd.date_range(start='2018', periods=len(df['Label'].values), freq='T')
    idx.min(), idx.max()
    df.set_index(idx, inplace=True)
    
    df = df.resample(window_size).mean()
    df['Label'] = df['Label'].astype(int)
    
    df['index'] = np.arange(1,len(df['Label'].values)+1,1) 
    df['index'] = df['index'].astype(int)
    df.set_index('index', inplace=True)
    
    return(df)
