from typing import List, NamedTuple

from parsers.naturallanguage.graphtextrepr import GraphTextRepresentation

TextSummary = NamedTuple('TextSummary', [('summary_sentence', List[str]), ('main_text', GraphTextRepresentation)])



    