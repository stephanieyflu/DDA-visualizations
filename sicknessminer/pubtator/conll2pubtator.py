#  This file is part of SicknessMiner.
#
#  SicknessMiner is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  SicknessMiner is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with SicknessMiner. If not, see <https://www.gnu.org/licenses/>.

"""
Converts a CoNLL file to PubTator format.
"""

import collections.abc
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional, List, Iterator

from sicknessminer.pubtator.utils import Mention

WHITESPACE_REGEX = re.compile("\\s")
PUBTATOR_SEPARATOR = "|"
PUBTATOR_END_OF_FIELD = "\n"
PUBTATOR_TITLE_KEY = "t"
PUBTATOR_ABSTRACT_KEY = "a"


class Parameters:
    """
    Represents the parameters of the state machine.
    """

    def __init__(self, output, to_print=False, incremental=False,
                 override_entity=None):
        self.to_print = to_print
        self.incremental = incremental
        self._output_identifier = -1
        if output is None:
            self.writer = None
        else:
            if isinstance(output, str):
                self.writer = open(output, "w")
            else:
                self.writer = output
        self.override_entity = override_entity
        self._initialize_values()

    def reset(self):
        """
        Resets the parameters.
        """
        self._initialize_values()

    def _initialize_values(self):
        self.offset_position = 0
        self.previous_identifier = ""
        self.current_identifier = ""
        self.previous_value = PUBTATOR_END_OF_FIELD
        self.mentions: List[Mention] = []
        if self.incremental:
            self._output_identifier += 1

    @property
    def output_identifier(self):
        """
        Gets the output identifier.

        :return: the output identifier
        :rtype: str
        """
        if self.incremental:
            return str(self._output_identifier)
        else:
            return self.previous_identifier

    def close(self):
        """
        Closes the output file.
        """
        if self.writer is not None:
            self.writer.close()

    def write(self, value):
        """
        Writes the value to the output.

        :param value: the value
        :type value: str
        """
        if self.writer is None:
            return

        self.writer.write(value)
        if self.to_print:
            print(value, end="", sep="")

    # noinspection PyAttributeOutsideInit
    def assert_identifier(self):
        """
        Asserts that the current identifier is the first identifier of the
        read article or it is equal to the previous identifier.
        """
        assert \
            self.previous_identifier == "" or \
            self.current_identifier == self.previous_identifier, \
            f"Identifier mismatch: expected {self.previous_identifier}, " \
            f"found {self.current_identifier}"
        self.previous_identifier = self.current_identifier
        self.current_identifier = ""

    def new_mention(self, mention_type):
        """
        Creates a new mention with type.

        :param mention_type: the mention type
        :type mention_type: str
        :return: the new mention
        :rtype: Mention
        """
        return Mention(self.output_identifier,
                       self.offset_position, -1, "", mention_type, None)

    def append_mention(self, mention):
        """
        Appends the mention to the list of mentions.

        :param mention: the mention
        :type mention: Mention
        """
        self.mentions.append(mention)

    def increase_offset(self, value=1):
        """
        Increases the offset by `value`.
        """
        self.offset_position += value

    def end_of_article(self, current_value):
        """
        Checks if it is the end of the article.

        :param current_value: the current value
        :type current_value: str
        :return: `True`, if it is the end of the article; otherwise, `False`.
        :rtype: bool
        """
        return \
            current_value == self.previous_value \
            and current_value == PUBTATOR_END_OF_FIELD

    def write_mentions(self):
        """
        Writes the mentions to the output file.
        """
        for mention in self.mentions:
            if self.override_entity is not None:
                mention = Mention(
                    mention.article_identifier, mention.begin_index,
                    mention.end_index, mention.text, self.override_entity,
                    mention.entity_identifier)
            self.write(str(mention))
            self.write("\n")
        self.write("\n")


class EntityType(Enum):
    """
    An entity type.
    """
    BEGINNING = "B"
    INSIDE = "I"
    OUTSIDE = "O"


class NamedEntity:
    """
    A named entity.
    """

    def __init__(self, entity_type: EntityType, value: Optional[str] = None,
                 line_number: Optional[int] = None):
        self.type: EntityType = entity_type
        self.value: Optional[str] = value
        self.line_number = line_number

    def __repr__(self):
        message = ""
        if self.line_number is not None:
            message = f"{self.line_number}: "
        if self.type == EntityType.OUTSIDE:
            return message + EntityType.OUTSIDE.value
        else:
            return message + f"{self.type.value}-{self.value}"


class ConllIterator(Iterator[Tuple[str, NamedEntity]]):
    """
    Iterates over a CoNLL file
    """

    def __init__(self, line_iterator, value_index=-1):
        if isinstance(line_iterator, str):
            self.line_iterator = open(line_iterator)
        else:
            self.line_iterator = line_iterator
        self.value_index = value_index
        self.line_number = 0

    def __next__(self) -> Tuple[str, NamedEntity]:
        line: str = ""
        while line == "":
            line = next(self.line_iterator).strip()
            self.line_number += 1
        fields = line.split()
        text = fields[0]
        value = fields[self.value_index]
        entity_type: EntityType = EntityType(value[0])
        named_entity_value = None
        if entity_type != EntityType.OUTSIDE:
            named_entity_value = value[2:]
        named_entity = NamedEntity(
            entity_type, named_entity_value, self.line_number)
        return text, named_entity


class State(ABC):
    """
    Implements a state of the state machine.
    """

    def __init__(self, parameters, next_state):
        """
        Initializes the state.

        :param parameters: the parameters
        :type parameters: Parameters
        :param next_state: the next state
        :type next_state: State
        """
        self.parameters = parameters
        self.next_state = next_state

    @abstractmethod
    def perform_operation(self, iterator):
        """
        Performs the operation of the state.

        :param iterator: the iterator of chars
        :type iterator: collections.abc.Iterator[str]
        :return: the new state
        :rtype: State
        """
        pass

    def is_final_state(self):
        """
        Returns true if the state is final.

        :return: `True` if the state is final; otherwise, `False`.
        :rtype: bool
        """
        return False

    # def reset_parameters(self):
    #     """
    #     Resets the parameters to its initial values.
    #     """
    #     self.parameters.reset()


class IdentifierState(State):
    """
    The identifier state.
    """

    def __init__(self, parameters, next_state, end_state):
        super().__init__(parameters, next_state)
        self.end_state = end_state

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, iterator):
        try:
            value = next(iterator)
            # Skip space characters before article
            while WHITESPACE_REGEX.match(value):
                value = next(iterator)

            identifier = ""
            while value != PUBTATOR_SEPARATOR:
                identifier += value
                # self.parameters.write(value)
                # self.parameters.current_identifier += value
                value = next(iterator)
            self.parameters.current_identifier = identifier
            self.parameters.assert_identifier()
            self.parameters.write(self.parameters.output_identifier)
            self.parameters.write(value)
        except StopIteration:
            return self.end_state

        return self.next_state


class SeparatorState(State):
    """
    The separator state.
    """

    def __init__(self, parameters, state_key, next_state):
        super().__init__(parameters, next_state)
        self.state_key = state_key

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, iterator):
        read_key = ""
        value = next(iterator)
        while value != PUBTATOR_SEPARATOR:
            read_key += value
            value = next(iterator)
        assert read_key == self.state_key, \
            f"Key mismatch: expected {self.state_key}, found {read_key}"
        self.parameters.write(read_key)
        self.parameters.write(value)

        return self.next_state


class ValueState(State):
    """
    The state to read the value of the article.
    """

    def __init__(self, parameters,
                 conll_iterator: Iterator[Tuple[str, NamedEntity]], next_state):
        super().__init__(parameters, next_state)
        self.conll_iterator = conll_iterator

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, iterator):
        mention: Optional[Mention] = None
        mention_text = ""
        value = next(iterator)
        while value != PUBTATOR_END_OF_FIELD:
            conll_name, conll_type = next(self.conll_iterator)
            if conll_type.type == EntityType.BEGINNING:
                # Beginning of a new mention
                if mention is not None:
                    # Flush current mention
                    mention.text = mention_text
                    mention.end_index = self.parameters.offset_position
                    self.parameters.append_mention(mention)
                # Starts a new mention
                mention_text = ""
                mention = self.parameters.new_mention(conll_type.value)
            elif conll_type.type == EntityType.INSIDE:
                # Inside of a mention
                if mention is not None:
                    # Checks if this mention is of the same type as current
                    # mention
                    if mention.entity_type != conll_type.value:
                        # It is not the same type, flush the current mention
                        # and skip this mention, since it has not beginning
                        mention.text = mention_text
                        mention.end_index = self.parameters.offset_position
                        self.parameters.append_mention(mention)
                        mention = None
                        mention_text = ""
            else:
                # It is not a mention: conll_type.type == EntityType.OUTSIDE
                if mention is not None:
                    # Flush current mention, if exists
                    mention.text = mention_text
                    mention.end_index = self.parameters.offset_position
                    self.parameters.append_mention(mention)
                    mention = None
            for char in conll_name:
                while WHITESPACE_REGEX.match(value):
                    self.parameters.write(value)
                    if mention_text != "":
                        mention_text += value
                    elif mention is not None:
                        mention.begin_index += 1
                    self.parameters.increase_offset()
                    value = next(iterator)
                assert value == char, f"Value mismatch: expected {value}, " \
                                      f"found {char} at CoNLL line number: " \
                                      f"\t{conll_type.line_number}"
                self.parameters.write(value)
                mention_text += value
                self.parameters.increase_offset()
                value = next(iterator)
        if mention is not None:
            # Flush current mention, if exists
            mention.text = mention_text
            mention.end_index = self.parameters.offset_position
            self.parameters.append_mention(mention)
        self.parameters.write(value)
        self.parameters.increase_offset()

        return self.next_state


class MentionState(State):
    """
    The state to consume the mentions of the file, if any.
    """

    def __init__(self, parameters, next_state, reset=True):
        super().__init__(parameters, next_state)
        self.reset = reset

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, iterator):
        try:
            value = next(iterator)
            while not self.parameters.end_of_article(value):
                self.parameters.previous_value = value
                value = next(iterator)
        except StopIteration:
            pass
        self.parameters.write_mentions()
        if self.reset:
            self.parameters.reset()

        return self.next_state


class FinalState(State):
    """
    The final state.
    """

    def __init__(self, parameters):
        super().__init__(parameters, self)

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, iterator):
        return self

    # noinspection PyMissingOrEmptyDocstring
    def is_final_state(self):
        return True


def build_state_machine(conll_iterator, output_path,
                        to_print=False, incremental=False,
                        override_entity=None, reset=True):
    """
    Builds the state machine.

    :param conll_iterator: the iterator of the CoNLL file.
    :type conll_iterator: ConllIterator
    :param output_path: the output filepath
    :type output_path: str or TextIO or None
    :param to_print: If set, the output will also be printed in the standard
    output
    :type to_print: bool
    :param incremental: If set, the output will use an incremental id for each
    paper, instead of the id found in the PubTator file.
    :type incremental: bool
    :return: the initial state of the machine
    :rtype: State
    """
    # Creates the parameters
    parameters = Parameters(
        output_path, to_print=to_print, incremental=incremental,
        override_entity=override_entity)

    # Creates the final state
    end_state = FinalState(parameters)

    # Creates the mention state
    # Uses the end_state as a placeholder while the first state is not ready
    mentions = MentionState(parameters, end_state, reset=reset)

    # Creates the abstract states
    abstract_value = ValueState(parameters, conll_iterator, mentions)
    abstract_separator = SeparatorState(
        parameters, PUBTATOR_ABSTRACT_KEY, abstract_value)
    abstract_identifier = IdentifierState(
        parameters, abstract_separator, end_state)

    # Creates the title states
    title_value = ValueState(parameters, conll_iterator, abstract_identifier)
    title_separator = SeparatorState(
        parameters, PUBTATOR_TITLE_KEY, title_value)
    title_identifier = IdentifierState(parameters, title_separator, end_state)

    # Replaces the end_state placeholder with the first state
    mentions.next_state = title_identifier

    return title_identifier

# def main():
#     """
#     The main method.
#     """
#     # parser = create_arguments_parser()
#     arguments = parser.parse_args()
#
#     input_conll = arguments.files[0]
#     input_pubtator = arguments.files[1]
#     output_path = arguments.files[2]
#
#     conll_iterator = ConllIterator(input_conll)
#     # pubtator = open(input_pubtator)
#     # char_iterator = itertools.chain.from_iterable(pubtator)
#     # pubtator_iterator = PubTatorIterator(input_pubtator)
#     # map(lambda x: str(x), pubtator_iterator)
#     char_iterator = itertools.chain.from_iterable(
#         map(lambda x: str(x), pubtator_iterator))
#
#     state = build_state_machine(conll_iterator, output_path,
#                                 to_print=arguments.to_print,
#                                 incremental=arguments.incremental,
#                                 override_entity=arguments.overrideEntity)
#
#     while not state.is_final_state():
#         state = state.perform_operation(char_iterator)
#
#     state.parameters.close()
