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
Converts a NormCo file to the PubTator format.
"""

import sys
from typing import List, Optional, Iterator, Dict

from sicknessminer.pubtator.utils import PubTatorEntry, Mention, \
    PubTatorIterator

HEADER = "ids\tmentions\tspans\tpmid"


class NormCoEntry:
    """
    Represents a NormCo entry.
    """

    def __init__(self, identifier: str, mentions: List[Mention]):
        self.identifier = identifier
        self.mentions = mentions

    def __repr__(self):
        mesh_ids = []
        str_mentions = []
        spans = []
        pmid = self.mentions[0].article_identifier

        for mention in self.mentions:
            mesh_ids.append(mention.entity_identifier)
            str_mentions.append(mention.text)
            spans.append(f"{mention.begin_index} {mention.end_index}")

        message = "|".join(mesh_ids)
        message += "\t"
        message += "|".join(str_mentions)
        message += "\t"
        message += "|".join(spans)
        message += "\t"
        message += pmid
        message += "\n"

        return message


def read_normco_entry(input_stream: Iterator[str],
                      entity_type) -> Optional[PubTatorEntry]:
    """
    Reads a NormCo entry from the `input_stream`, if exists.

    :param input_stream: the stream
    :type input_stream: Iterator[str]
    :param entity_type: the type of the entity
    :type entity_type: str
    :return: the PubTator entry, if exists; otherwise, `None`.
    :rtype: Optional[NormCoEntry]
    """
    entry = None
    try:
        line: str = next(input_stream)
        line = line.strip()
        while line == "" or HEADER in line:
            line = next(input_stream)
            line = line.strip()

        mentions = []

        fields = line.split("\t")
        identifier = fields[-1]
        mention_ids = fields[0].split("|")
        mention_texts = fields[1].split("|")
        mention_spans = fields[2].split("|")

        for entity_id, mention_text, mention_span in \
                zip(mention_ids, mention_texts, mention_spans):
            indices = mention_span.split()
            mentions.append(Mention(
                identifier, int(indices[0]), int(indices[1]),
                mention_text, entity_type, entity_id
            ))

        entry = NormCoEntry(identifier, mentions)
    except StopIteration:
        pass
    return entry


def merge_mentions(pubtator_entry, normco_mentions, override_type):
    """
    Merges the PubTator mentions with the NormCo mentions. It assumes that
    the PubTator mention contains all the NormCo mentions.

    :param pubtator_entry: the PubTatorEntry
    :type pubtator_entry: PubTatorEntry
    :param normco_mentions: the NormCo mentions
    :type normco_mentions: List[utils.pubtator.Mention]
    :param override_type: if set, replaces the PubTator mention entity type
    by the NormCo entity type
    :type override_type: bool
    :return: the merged mentions
    :rtype: List[utils.pubtator.Mention]
    """
    pubtator_mentions = pubtator_entry.mentions
    merged_mentions = []
    normco_it = iter(normco_mentions)
    normco_mention = next(normco_it)
    try:
        for mention in pubtator_mentions:
            if normco_mention is None:
                merged_mentions.append(mention)
            elif mention.begin_index == normco_mention.begin_index \
                    and mention.end_index == normco_mention.end_index:
                assert \
                    mention.article_identifier == \
                    normco_mention.article_identifier, \
                    f"Identifier mismatch: expected " \
                    f"{mention.article_identifier}, " \
                    f"found {normco_mention.article_identifier}"
                mention_clean_text = mention.text.replace("\"", "")
                normco_clean_text = normco_mention.text.replace("\"", "")
                assert mention_clean_text == normco_clean_text, \
                    f"Text mismatch: expected {mention_clean_text}, " \
                    f"found {normco_clean_text}"
                merged = Mention(
                    mention.article_identifier, mention.begin_index,
                    mention.end_index, mention.text,
                    normco_mention.entity_type, normco_mention.entity_identifier
                )
                if not override_type:
                    assert mention.entity_type == normco_mention.entity_type, \
                        f"Entity Type mismatch: expected " \
                        f"{mention.entity_type}, " \
                        f"found {normco_mention.entity_type}"
                merged_mentions.append(merged)
                try:
                    normco_mention = next(normco_it)
                except StopIteration:
                    normco_mention = None
            else:
                merged_mentions.append(mention)
    except AssertionError as e:
        print(f"Error verifying mention for:\t{pubtator_entry.identifier}, {e}",
              file=sys.stderr)
        return pubtator_entry.mentions

    return merged_mentions


def get_normco_entries(normco_iterator, entity_type) -> Dict[str, NormCoEntry]:
    """
    Reads all the entries from a NormCo iterator into a dictionary.

    :param normco_iterator: the NormCo iterator
    :type normco_iterator: Iterator[str]
    :param entity_type: the entity type
    :type entity_type: str
    :return: the NormCo entries
    :rtype: Dict[str, NormCoEntry]
    """
    entries = []
    while True:
        try:
            entry = read_normco_entry(normco_iterator, entity_type)
            if entry is not None:
                entries.append(entry)
            else:
                break
        except StopIteration:
            break

    return dict(map(lambda x: (x.identifier, x), entries))


def main(normco_iterator, pubtator_path, output_path, entity_type,
         override_type):
    """The main function."""
    writer = open(output_path, "w")

    pubtator_iterator = PubTatorIterator(pubtator_path)
    normco_entries = get_normco_entries(normco_iterator, entity_type)
    for pubtator_entry in pubtator_iterator:
        identifier = pubtator_entry.identifier
        normco_entry = normco_entries.get(identifier)
        if normco_entry is not None:
            mentions = merge_mentions(
                pubtator_entry, normco_entry.mentions, override_type)
            pubtator_entry.mentions = mentions

        writer.write(str(pubtator_entry))

    writer.close()
