_NOTE: Python code was not used in the paper due reasons explained there. However code is still made available for public use_

To parse the data from Barone _et al_.

First, clone the repository:
```
git clone https://github.com/EdinburghNLP/code-docstring-corpus.git
```

Then run `loadbarone.py` for each fold run:
```
$ python loadbarone.py path/to/repo/code-docstring-corpus/parallel-corpus/{data_ps.declbodies.train,data_ps.descriptions.train} /folder/to/out/data_
```
repeat by replacing "train" with "valid" and "test". This creates the dataset from code to docstring.

To create the dataset for code to function name, use the following instead:
```
$ python loadbarone.py --task-type func-name path/to/repo/code-docstring-corpus/parallel-corpus/{data_ps.bodies.train,data_ps.declarations.train} /folder/to/out
```