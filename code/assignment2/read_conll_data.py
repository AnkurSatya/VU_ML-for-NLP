from nltk.corpus.reader.conll import ConllCorpusReader, LazyConcatenation, LazyMap


###Taken from [https://stackoverflow.com/questions/46872083/ne-tags-in-nltk-conllcorpusreader] [14/11/2022]
class betterConllReader(ConllCorpusReader):

    def iob_words(self, fileids=None, tagset=None, column="ne"):
        """
        :return: a list of word/tag/IOB tuples
        :rtype: list(tuple)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.WORDS, self.POS, self.CHUNK, self.NE)
        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset, column)
        return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))

    def _get_iob_words(self, grid, tagset=None, column="ne"):
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap['words']), pos_tags,
                   self._get_column(grid, self._colmap[column])))

###

def read_data(root_filename, datafiles, column_types):
    # data_stream = pyconll.load_from_file(filename)
    # data = parse(data_stream)

    data = betterConllReader(root_filename, datafiles, columntypes=column_types)
    return data


