import io
# import pyconll
import nltk
from nltk.corpus.reader.conll import ConllCorpusReader

def read_data(root_filename, datafiles, column_types):
    # data_stream = pyconll.load_from_file(filename)
    # data = parse(data_stream)

    data = ConllCorpusReader(root_filename, datafiles, columntypes=column_types)
    # print(data.sents()[0])
    print(nltk.ne_chunk(data.sents()))


def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file
    
    :param inputfile: the path to the conll file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type annotationcolumn: string
    :type delimiter: string
    :returns: the annotations as a list
    '''
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    conll_input = pd.read_csv(inputfile, sep=delimiter, error_bad_lines=False)
    annotations = conll_input[annotationcolumn].tolist()
    return annotations


def get_class_labels_distribution():
    pass


if __name__ == "__main__":
    nltk.download('words')
    nltk.download('maxent_ne_chunker')
    root_filename = "../../data/"
    datafiles = ["conll2003.train.conll"]
    column_types = ["words", "pos", "chunk", "ne"]
    read_data(root_filename, datafiles, column_types)