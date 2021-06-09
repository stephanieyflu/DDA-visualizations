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
Useful method to deal with PubTator files.
"""

import itertools
from typing import List, Iterator, Optional, TextIO


class Mention:
    """
    Defines a mention in the PubTator text.
    """

    def __init__(self,
                 article_identifier: str,
                 begin_index: int,
                 end_index: int,
                 text: str,
                 entity_type: str,
                 entity_identifier: Optional[str] = None):
        self.article_identifier = article_identifier
        "The article identifier"

        self.begin_index = begin_index
        "The index of the beginning of the mention"

        self.end_index = end_index
        "The index of the end of the mention"

        self.text = text
        "The text of the mention"

        self.entity_type = entity_type
        "The type of the entity in the mention"

        self.entity_identifier = entity_identifier
        "The identifier of the entity in the mention"

    def __repr__(self):
        string = f"{self.article_identifier}" \
                 f"\t{self.begin_index}" \
                 f"\t{self.end_index}" \
                 f"\t{self.text}" \
                 f"\t{self.entity_type}"
        if self.entity_identifier is not None:
            string += f"\t{self.entity_identifier}"

        return string

    def keys(self):
        """
        Returns the keys.

        :return: a tuple of all the fields
        :rtype: tuple
        """
        return (
            self.article_identifier, self.begin_index, self.end_index,
            self.text, self.entity_type, self.entity_identifier
        )

    def __eq__(self, other):
        if not isinstance(other, Mention):
            return False

        return self.keys() == other.keys()


class PubTatorEntry:
    """
    Represents a PubTator entry.
    """

    def __init__(self, identifier: str, title: str, abstract: str,
                 mentions: List[Mention]):
        self.identifier = identifier
        self.title = title
        self.abstract = abstract
        self.mentions = mentions

    def __repr__(self):
        message = f"{self.identifier.strip()}|t|{self.title.strip()}\n"
        message += f"{self.identifier.strip()}|a|{self.abstract.strip()}\n"
        for mention in self.mentions:
            message += f"{str(mention).strip()}\n"
        message += "\n"

        return message

    def to_string_without_mentions(self):
        message = f"{self.identifier.strip()}|t|{self.title.strip()}\n"
        message += f"{self.identifier.strip()}|a|{self.abstract.strip()}\n\n"
        return message

    def keys(self):
        """
        Returns the keys.

        :return: a tuple of all the fields
        :rtype: tuple
        """
        return self.identifier, self.title, self.abstract, tuple(self.mentions)

    def __eq__(self, other):
        if not isinstance(other, PubTatorEntry):
            return False

        return self.keys() == other.keys()

    def replace_identifier(self, identifier):
        self.identifier = identifier
        for mention in self.mentions:
            mention.article_identifier = identifier

    def __hash__(self):
        return hash(self.identifier)


class PubTatorIterator(Iterator[PubTatorEntry]):
    """
    Iterates over a PubTator file.
    """

    def __init__(self, file_paths):
        """
        Creates an iterator of PubTatorEntry(s), read from a list of PubTator
        files.

        :param file_paths: The paths to the PubTator files
        :type file_paths: List[str] or str
        """
        if isinstance(file_paths, list) or isinstance(file_paths, tuple):
            if len(file_paths) > 1:
                self.pubtator_file = \
                    itertools.chain.from_iterable(
                        map(lambda x: open(x), file_paths))
            else:
                self.pubtator_file = open(file_paths[0])
        else:
            self.pubtator_file = open(file_paths)

    def __next__(self) -> PubTatorEntry:
        entry = read_pubtator_entry(self.pubtator_file)
        if entry is None:
            raise StopIteration
        return entry


def read_pubtator_entry(input_stream: TextIO) -> Optional[PubTatorEntry]:
    """
    Reads a PubTator entry from the `input_stream`, if exists.

    :param input_stream: the stream
    :type input_stream: TextIO
    :return: the PubTator entry, if exists; otherwise, `None`.
    :rtype: Optional[PubTatorEntry]
    """
    entry = None
    try:
        line: str = next(input_stream)
        while line.strip() == "":
            line = next(input_stream)

        fields = line.strip().split("|")
        identifier = fields[0]
        assert fields[1] == "t", f"Expected t, found {fields[1]}"
        title = fields[2]
        line = next(input_stream)
        fields = line.strip().split("|")
        assert identifier == fields[0], \
            f"Expected identifier {identifier}, found {fields[0]}"
        assert fields[1] == "a", f"Expected a, found {fields[1]}"
        abstract = fields[2]
        mentions = []
        entry = PubTatorEntry(identifier, title, abstract, mentions)
        line = next(input_stream)
        while line.strip() != "":
            fields = line.strip().split("\t")
            assert identifier == fields[0], \
                f"Expected identifier {identifier}, found {fields[0]}"
            if len(fields) > 4:
                mentions.append(Mention(
                    fields[0],
                    int(fields[1]),
                    int(fields[2]),
                    fields[3],
                    fields[4],
                    fields[5] if len(fields) > 5 else None
                ))
            line = next(input_stream)
    except StopIteration:
        pass
    return entry


class RawTextToPubTator(Iterator[PubTatorEntry]):
    """
    Iterates over a raw text files and generates PubTator entries.
    """

    def __init__(self, file_paths, initial_index=0):
        """
        Creates an iterator of PubTatorEntry(s), read from a list of raw
        text files.

        :param file_paths: The paths to the PubTator files
        :type file_paths: List[str] or str
        :param initial_index: the initial index
        :type initial_index: int
        """
        if isinstance(file_paths, list) or isinstance(file_paths, tuple):
            if len(file_paths) > 1:
                self.input_files = \
                    itertools.chain.from_iterable(
                        map(lambda x: open(x), file_paths))
            else:
                self.input_files = open(file_paths[0])
        else:
            self.input_files = open(file_paths)
        self.index = initial_index

    def __next__(self) -> PubTatorEntry:
        current_value = ""
        try:
            for line in self.input_files:
                line = line.strip()
                if line == "":
                    if current_value != "":
                        entry = PubTatorEntry(
                            str(self.index), "", current_value.strip(), [])
                        self.index += 1
                        return entry
                    continue
                current_value += line + " "
        except StopIteration as e:
            if current_value == "":
                raise e
            else:
                entry = PubTatorEntry(
                    str(self.index), "", current_value.strip(), [])
                self.index += 1
                return entry

        if current_value != "":
            entry = PubTatorEntry(
                str(self.index), "", current_value.strip(), [])
            self.index += 1
            return entry
        else:
            raise StopIteration()
