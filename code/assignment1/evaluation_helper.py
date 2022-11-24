import sys
import pandas as pd
import numpy as np
# see tips & tricks on using defaultdict (remove when you do not use it)
from collections import defaultdict, Counter
# module for verifying output
from nose.tools import assert_equal

def get_gold_and_system_annotations(prediction_file):
    
    with open('prediction.txt') as f:
        lines = f.readlines()
        gold_annotations, system_annotations = [], []
        for line in lines:
            line = line.rstrip("\n")
            line = line.split("\t")
            gold_annotations.append(line[-2])
            system_annotations.append(line[-1])
            
    return gold_annotations, system_annotations

def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output
    
    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations
    
    :returns: a countainer providing the counts for each predicted and gold class pair
    '''
    
    # TIP on how to get the counts for each class
    # https://stackoverflow.com/questions/49393683/how-to-count-items-in-a-nested-dictionary, last accessed 22.10.2020
    evaluation_counts = defaultdict(Counter)
    
#     pure_class_gold = [re.search("(.*\-)?(.*)", elm).groups()[1] for elm in goldannotations]
#     pure_class_machine = [re.search("(.*\-)?(.*)", elm).groups()[1] for elm in machineannotations]
    
    unique_class = list(set(goldannotations).union(machineannotations))
    unique_class.sort()
    
    for elm in unique_class:
        evaluation_counts[elm] = {e: 0 for e in unique_class}
    
    for gold_elm, machine_elm in zip(goldannotations, machineannotations):
        evaluation_counts[gold_elm][machine_elm] +=1
    
    return evaluation_counts
    
    
def get_precision_recall_metrics(evaluation_counts):
    '''
    This function calculates the distribution of prediction by the system against the gold results
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns: a countainer providing the counts for each predicted and gold class pair in an array form.
    '''
    data = []
    for i in evaluation_counts.keys():
        data.append(list(evaluation_counts[i].values()))
    
    data = np.array(data)
    return data
    
def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns the precision, recall and f-score of each class in a container
    '''
    
    # TIP: you may want to write a separate function that provides an overview of true positives, false positives and false negatives
    #      for each class based on the outcome of obtain counts
    # YOUR CODE HERE (and remove statement below)
    
    entities = evaluation_counts.keys()
    result_dict = {e:{} for e in entities}
    metrics = get_precision_recall_metrics(evaluation_counts)
    
    for i, ent in enumerate(entities):
        tp = metrics[i][i]
        fn = sum(metrics[i, :]) - tp
        fp = sum(metrics[:, i]) - tp
        
        if tp+fp == 0:
            precision = np.nan
        else:
            precision = round(1.0*tp/(tp+fp), 5)

        if tp+fn == 0:
            recall = np.nan    
        else:
            recall = round(1.0*tp/(tp+fn), 5)
        
        if precision is np.nan or recall is np.nan:
            f_score = np.nan
        else:
            f_score = round(2.0 * precision * recall/(precision + recall), 5)
        
        result_dict[ent]["precision"] = precision
        result_dict[ent]["recall"] = recall
        result_dict[ent]["f-score"] = f_score
        
    return result_dict
            

def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :prints out a confusion matrix
    '''
    
    # TIP: provide_output_tables does something similar, but those tables are assuming one additional nested layer
    #      your solution can thus be a simpler version of the one provided in provide_output_tables below
    
    # YOUR CODE HERE (and remove statement below)
    
    data = []
    columns = ["Gold Tag " + u'\u2193' + " | Machine Tag " + u'\u2192']
    for i in evaluation_counts.keys():
        columns.append(i)
        data.append([i])
        data[-1].extend(evaluation_counts[i].values())
        
    confusion_matrix = pd.DataFrame(data, columns=columns)
    # print(confusion_matrix)


def carry_out_evaluation(gold_annotations, systemfile, systemcolumn, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns evaluation information for this specific system
    '''
    system_annotations = extract_annotations(systemfile, systemcolumn, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)
    
    return evaluation_outcome


def carry_out_evaluation(gold_annotations, system_annotations, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns evaluation information for this specific system
    '''
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)
    
    return evaluation_outcome


def run_evaluations(predicted_file):
    '''
    Carry out standard evaluation for one or more system outputs
    
    :param goldfile: path to file with goldstandard
    :param goldcolumn: indicator of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: integer
    :type systems: list (providing file name, information on tab with system output and system name for each element)
    
    :returns the evaluations for all systems
    '''
    evaluations = {}
    #not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations, system_annotations = get_gold_and_system_annotations(predicted_file)
    return carry_out_evaluation(gold_annotations, system_annotations)

    # for system in systems:
    #     sys_evaluation = carry_out_evaluation(gold_annotations, system_annotations, system[0], system[1])
    #     evaluations[system[2]] = sys_evaluation
    # return evaluations