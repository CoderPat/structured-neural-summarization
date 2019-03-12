from collections import defaultdict
from io import StringIO
from itertools import chain
from typing import Dict, Optional, List, NamedTuple, Set, Union, Tuple

Token = NamedTuple('Token', [
    ('word', str),
    ('lemma', str),
    ('pos_tag', str)
])

DependencyEdge = NamedTuple('DependencyEdge', [
    ('dependency_type', str),
    ('sentence_idx', int),
    ('from_idx', int),
    ('to_idx', int)
])


Entity = NamedTuple('Entity', [
    ('entity_type', str),
    ('sentence_idx', int),
    ('start_idx', int),
    ('end_idx', int)
])


class GraphTextRepresentation:
    def __init__(self, tokenized_text: List[List[Token]], provenance: str)-> None:
        self.__provenance = provenance
        self.__tokenized_text = tokenized_text
        self.__dependency_edges = []  # type: List[DependencyEdge]
        self.__entities = defaultdict(set)  # type: Dict[int, Set[Entity]]
        self.__coreferences = []  # type: List[Tuple[Union[Entity, Tuple[int, int]], Union[Entity, Tuple[int, int]]]]

    @property
    def text(self)-> str:
        return ' '.join(' '.join(token.word for token in sentence) for sentence in self.__tokenized_text)

    @property
    def words(self) -> List[str]:
        return list(chain(*((token.word for token in sentence) for sentence in self.__tokenized_text)))

    @property
    def provenance(self)-> str:
        return self.__provenance

    def add_dependency_edge(self, edge: DependencyEdge):
        assert 0 <= edge.sentence_idx < len(self.__tokenized_text)
        assert 0 <= edge.from_idx < len(self.__tokenized_text[edge.sentence_idx])
        assert 0 <= edge.to_idx < len(self.__tokenized_text[edge.sentence_idx])
        self.__dependency_edges.append(edge)

    def add_entity(self, entity_type: str, sentence_idx: int, from_idx: int, to_idx: int)->None:
        assert 0 <= sentence_idx < len(self.__tokenized_text)
        assert 0 <= from_idx < to_idx <= len(self.__tokenized_text[sentence_idx])
        self.__entities[sentence_idx].add(
            Entity(entity_type, sentence_idx, from_idx, to_idx))

    def __entities_covered_by(self, sentence_idx: int, location_idx: int) -> Optional[Entity]:
        entities = [entity for entity in self.__entities[sentence_idx] if entity.start_idx <= location_idx < entity.end_idx]
        assert len(entities) <= 1
        if len(entities) == 0:
            return None
        return entities[0]

    def add_coreference(self, from_sentence_idx: int, from_head_idx: int, to_sentence_idx: int, to_head_idx: int)-> None:
        """
        Add a coreference across sentences. This assumes that all entities have been added first.
        """
        def get_coreference_location(sentence_idx: int, head_idx: int)-> Union[Entity, Tuple[int, int]]:
            underlying_entity = self.__entities_covered_by(sentence_idx, head_idx)
            return underlying_entity or (sentence_idx, head_idx)
        assert not (from_sentence_idx == to_sentence_idx and from_head_idx == to_head_idx)
        self.__coreferences.append((get_coreference_location(from_sentence_idx, from_head_idx), get_coreference_location(to_sentence_idx, to_head_idx)))

    def to_graph_object(self):
        node_labels = [] #  type: List[str]
        node_edges = []  # type: List[Tuple[str, int, int]]

        def add_node(node_label: str)->int:
            idx = len(node_labels)
            node_labels.append(node_label)
            return idx

        word_node_location = defaultdict(dict)  # type: Dict[int, Dict[int, int]]  # sentence_idx -> idx -> node_id

        for sentence_idx, sentence in enumerate(self.__tokenized_text):
            for token_idx, token in enumerate(sentence):
                node_idx = add_node(token.word)
                word_node_location[sentence_idx][token_idx] = node_idx
                if node_idx > 0:
                    node_edges.append(('next', node_idx - 1, node_idx))

        num_words = len(node_labels)

        # Add sentence hypernodes
        prev_sentence_node_idx = None
        for sentence_idx, sentence in enumerate(self.__tokenized_text):
            sentence_node_idx = add_node('Sentence')
            if prev_sentence_node_idx is not None:
                node_edges.append(('next', prev_sentence_node_idx, sentence_node_idx))

            for token_idx, token in enumerate(sentence):
                node_edges.append(('in', word_node_location[sentence_idx][token_idx], sentence_node_idx))
            prev_sentence_node_idx = sentence_node_idx

        # Add dependency edges
        for dep_edge in self.__dependency_edges:
            node_edges.append((dep_edge.dependency_type, word_node_location[dep_edge.sentence_idx][dep_edge.from_idx],
                               word_node_location[dep_edge.sentence_idx][dep_edge.to_idx]))

        # Add entities
        entities_idx = {}  # type: Dict[Entity, int]
        for sentence_idx, sentence_entities in self.__entities.items():
            for entity in sentence_entities:
                entity_idx = add_node('ENT:' + entity.entity_type)
                entities_idx[entity] = entity_idx
                for token_idx in range(entity.start_idx, entity.end_idx):
                    node_edges.append(('in', word_node_location[entity.sentence_idx][token_idx], entity_idx))

        # Add co-references
        def get_node_id(mention: Union[Entity, Tuple[int, int]]):
            if type(mention) is Entity:
                return entities_idx[mention]
            return word_node_location[mention[0]][mention[1]]
        for mention_from, mention_to in self.__coreferences:
            node_edges.append(('ref', get_node_id(mention_from), get_node_id(mention_to)))

        return {'node_labels': node_labels, 'edges': node_edges,
                'backbone_sequence': list(range(num_words)), 'provenance': self.provenance}


    def str_summary(self) -> str:
        with StringIO() as sb:
            sb.write('Text:' + self.text + '\n')

            sb.write('Dependency Edges\n')
            for depedge in self.__dependency_edges:
                sentence = self.__tokenized_text[depedge.sentence_idx]
                sb.write('"%s" --%s--> "%s" \n' % (sentence[depedge.from_idx].word, depedge.dependency_type, sentence[depedge.to_idx].word))

            sb.write('\n\nEntities\n')
            for sentence_idx, entities in self.__entities.items():
                for entity in entities:
                    sb.write('At sentence %s, entity %s with text "%s"\n' % (sentence_idx, entity.entity_type,
                                                                           ' '.join(t.word for t in self.__tokenized_text[sentence_idx][entity.start_idx: entity.end_idx])))

            def mention_text(mention_location: Union[Tuple, Entity])->str:
                if type(mention_location) is Entity:
                    sent_idx = mention_location.sentence_idx
                    return '%s("%s")@Sentence %s' % (mention_location.entity_type, ' '.join(t.word for t in self.__tokenized_text[sent_idx][mention_location.start_idx: mention_location.end_idx]), sent_idx)
                else:
                    return '"%s"@Sentence %s' % (self.__tokenized_text[mention_location[0]][mention_location[1]].word, mention_location[0])

            sb.write('\n\nCoreferences\n')
            for (mention_from, mention_to) in self.__coreferences:
                sb.write(mention_text(mention_from) + '-->' + mention_text(mention_to) + '\n')

            return sb.getvalue()

            

