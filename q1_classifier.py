import random
import pickle as pkl
import argparse
import csv
import numpy as np
import sys
import os
import pandas as pd
from scipy.stats import chisquare


sys.setrecursionlimit(10000)

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)


num_feats = 274
num_nodes = 0

features = pd.read_csv('featnames.csv', header=None, delim_whitespace= True)
#print(attr_list)


# Get entropy for an attribute selectio, this is based on huffman entropy formula given by:
# H(s) = -1 * Summation of(pi * log(pi))

def get_entropy(train_df, y_train, attr_idx):
    total_entropy = 0.0
    attr_df = pd.DataFrame()
    attr_df['attribute_val'] = train_df[attr_idx]
    attr_df["Output"] = y_train

    unique_attr_values = attr_df['attribute_val'].unique()

    for val in unique_attr_values:
        new_df = attr_df[attr_df['attribute_val'] == val]

        total_instances = new_df.shape[0]

        pos_instances = float(new_df["Output"].sum())
        neg_instances = total_instances - pos_instances

        prob_attr_val = float(new_df.shape[0])/attr_df.shape[0]

        prob_pos = pos_instances/total_instances
        prob_neg = neg_instances/total_instances
        
        val1 = 0 if prob_pos == 0 else prob_pos * np.log2(prob_pos)
        val2 = 0 if prob_neg == 0 else prob_neg * np.log2(prob_neg)
        
        attr_entropy =  -1 * (val1 + val2)

        total_entropy += (prob_attr_val) * attr_entropy

    #print "Total entropy = " + str(total_entropy)
    return total_entropy



#This function will be used for terminating recusrsion when pValue at a partuicular 
#node falls below chisquare distribution pval
def stop_further_split(train_df, y_train, attr_idx, pValue):

    expected_freq = []
    observed_freq = []

    attr_df = pd.DataFrame()
    attr_df['attribute_val'] = train_df[attr_idx]
    attr_df["Output"] = y_train
    #print(attr_df.head(5))
    
    total = len(y_train)
    num_positives = (y_train[0] == 1).sum()
    num_negatives = total - num_positives

    prob_pos = float(num_positives)/total
    prob_neg = float(num_negatives)/total

    unique_attr_values = attr_df['attribute_val'].unique()
    #print(unique_attr_values)

    for val in unique_attr_values:
        new_df = attr_df[attr_df['attribute_val'] == val]

        total_instances = new_df.shape[0]

        obs_pos_instances = float(new_df["Output"].sum())
        obs_neg_instances = total_instances - obs_pos_instances

        expected_pos = float(prob_pos)*total_instances
        expected_neg = float(prob_neg)*total_instances

        expected_freq += [expected_pos, expected_neg] 
        observed_freq += [obs_pos_instances, obs_neg_instances]


    chiSquare, pval = chisquare(observed_freq, expected_freq)
    #print(observed_freq, expected_freq)
    return (True if pval <= pValue else False)


# This function will return the termainal nodes
def terminal_node(y_train):
    num_pos = len(y_train[y_train[0] == 1])
    num_neg = len(y_train[y_train[0] == 0])
    global num_nodes
    num_nodes += 1
    if num_pos >= num_neg:
        return TreeNode('T')
    else:
        return TreeNode('F')



# Iterate over all attribute which are still left for this current path in the tree and get the feature having minimum 
# entropy which means max Information gain (IG) 
def get_best_attribute(train_df, y_train, attr_list):
    best_attr = -1
    min_entropy = sys.maxsize
    #print "length = " + str(len(attr_list))
    for attr_idx in attr_list:
        cur_entropy = get_entropy(train_df, y_train, attr_idx)
        if cur_entropy < min_entropy:
            min_entropy = cur_entropy
            best_attr = attr_idx

    return best_attr
    

# This is the main function to recurse for ID3 algorithm.
def ID3_tree(train_df, y_train, pval, attr_list):

    global num_nodes 

    if len(y_train[y_train[0] == 1]) == len(y_train):
        num_nodes += 1
        return TreeNode('T')

    if len(y_train[y_train[0] == 0]) == len(y_train):
        num_nodes += 1
        return TreeNode('F')

    if(len(attr_list) == 0):
        return terminal_node(y_train)

    maxGainAttr = get_best_attribute(train_df, y_train, attr_list)

    if stop_further_split(train_df, y_train, maxGainAttr, pval):
        
        #print "Max Gain node = " + str(maxGainAttr)
        num_nodes += 1
        root = TreeNode(maxGainAttr + 1)
        attr_new = list(attr_list)
        attr_new.remove(maxGainAttr)
        
        # Recurse on the children now
        for i in range(1,len(root.nodes)+1):
            new_df = pd.DataFrame(train_df)
            #print(type(y_train[0]))
            new_df['Output'] = y_train[0]
            new_df = train_df[train_df[maxGainAttr] == i]
            
            y_train_new = pd.DataFrame()
            y_train_new[0] = new_df['Output']
            #print(new_df.columns)
            del new_df['Output']
            root.nodes[i-1] = ID3_tree(new_df, y_train_new, pval, attr_new)
    else:
        return terminal_node(y_train)

    return root

    
parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = float(args['p'])
train_file = args['f1']
train_output = args['f1'].split('.')[0]+ '_label.csv' #labels filename will be the same as training file name but with _label at the end
test_file = args['f2']
output_file = args['o']
tree_name = args['t']

train_df = pd.read_csv(train_file, header = None, delim_whitespace = True)
y_train = pd.read_csv(train_output, header = None)
test_df = pd.read_csv(test_file, header = None, delim_whitespace = True)

#print(pval)
# print(train_df.head(5))
# print(test_df.head(5))
#print(y_train.head(5))
#print(attributes.head())
#print(train_df[267])

attr_list = list(xrange(len(features)))
print("Training...")
root = ID3_tree(train_df, y_train, pval, attr_list)
root.save_tree(tree_name)


def compute_test_example(root, example):
    if root.data == 'F':
        return 0
    if root.data == 'T':
        return 1

    #print root.data
    next = int(example[int(root.data)-1])-1
    return compute_test_example(root.nodes[next], example)

print("Testing...")
Ypredict = []
#generate random labels
for i in range(0,len(test_df)): 
    val = compute_test_example(root, test_df[i:i+1])
    Ypredict.append([val])

with open(output_file, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print ("Num Nodes = " + str(num_nodes))
print("Output files generated")








