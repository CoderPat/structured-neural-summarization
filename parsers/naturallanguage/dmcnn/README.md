To download the file go to https://cs.nyu.edu/~kcho/DMQA/. Once the files have been split use Stanford NLP to get the XML representation: 

```bash
./corenlp.sh -annotators tokenize,ssplit,pos,lemma,ner,parse,depparse,coref -coref.algorithm neural -filelist path/to/filelist.txt outputFormat xml -outputDirectory /path/to/output/xml
```

Then to process the data run

```bash
python convert2graph.py /path/to/output/xml /path/to/summaries /path/to/output
```



