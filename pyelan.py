import os
from itertools import groupby, chain, pairwise
from lxml import etree
import numpy as np
import string
import random



def create_eaf_file(new_file, media_file, tier_id=None):
    annotation_doc = etree.Element("ANNOTATION_DOCUMENT")
    annotation_doc.set("AUTHOR", "")
    annotation_doc.set("DATE", "2022-01-01")
    annotation_doc.set("FORMAT", "3.0")
    annotation_doc.set("VERSION", "3.0")
    annotation_doc.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
                       "http://www.mpi.nl/tools/elan/EAFv3.0.xsd")

    header = etree.SubElement(annotation_doc, "HEADER")
    header.set("MEDIA_FILE", "")
    header.set("TIME_UNITS", "milliseconds")
    descriptor = etree.SubElement(header, "MEDIA_DESCRIPTOR")
    descriptor.set("MEDIA_URL", "file:///" + media_file)
    descriptor.set("MIME_TYPE", "video/mp4")
    property_a = etree.SubElement(header, "PROPERTY")
    property_a.set("NAME", "URN")
    property_a.text = "urn:nl-mpi-tools-elan-eaf:" + _generate_identifier()
    property_b = etree.SubElement(header, "PROPERTY")
    property_b.set("NAME", "lastUsedAnnotationId")
    property_b.text = "0"
    time_order = etree.SubElement(annotation_doc, "TIME_ORDER")

    if tier_id != None:
        tier = etree.SubElement(annotation_doc, "TIER")
        tier.set("LINGUISTIC_TYPE_REF", "default-lt")
        tier.set("TIER_ID", tier_id)
        linguistic = etree.SubElement(annotation_doc, "LINGUISTIC_TYPE")
        linguistic.set("GRAPHIC_REFERENCES", "false")
        linguistic.set("LINGUISTIC_TYPE_ID", "default-lt")
        linguistic.set("TIME_ALIGNABLE", "true")

    constraint_a = etree.SubElement(annotation_doc, "CONSTRAINT")
    constraint_a.set("DESCRIPTION", "Time subdivision of parent annotation\'s time interval, "
                                    "no time gaps allowed within this interval")
    constraint_a.set("STEREOTYPE", "Time_subdivision")
    constraint_b = etree.SubElement(annotation_doc, "CONSTRAINT")
    constraint_b.set("DESCRIPTION", "Symbolic subdivision of a parent annotation. "
                                    "Annotations refering to the same parent are ordered")
    constraint_b.set("STEREOTYPE", "Symbolic_Subdivision")
    constraint_c = etree.SubElement(annotation_doc, "CONSTRAINT")
    constraint_c.set("DESCRIPTION", "1-1 association with a parent annotation")
    constraint_c.set("STEREOTYPE", "Symbolic_Association")
    constraint_d = etree.SubElement(annotation_doc, "CONSTRAINT")
    constraint_d.set("DESCRIPTION", "Time alignable annotations within the parent annotation\'s "
                                    "time interval, gaps are allowed")
    constraint_d.set("STEREOTYPE", "Included_In")

    open(new_file, 'w').close()
    etree.indent(annotation_doc, space="    ")
    new_annotation_doc = etree.ElementTree(annotation_doc)
    new_annotation_doc.write(new_file, pretty_print=True, xml_declaration=True, encoding='utf-8')


def add_elan_tier(eaf_file, tier_id, time_slots, values=None, newfile=False):
    speaker = Speaker(eaf_file)
    new_annotations_info = _format_annotations_info(speaker, time_slots, values=values)

    last_anno_id = speaker.annotation_doc.find("HEADER").findall("PROPERTY")[-1]
    new_last_anno_id = etree.Element("PROPERTY")
    new_last_anno_id.set("NAME", "lastUsedAnnotationId")
    new_last_anno_id.text = str(int(last_anno_id.text) + len(time_slots.flatten()//2))
    speaker.annotation_doc.find("HEADER").replace(last_anno_id, new_last_anno_id)

    new_time_order = etree.Element("TIME_ORDER")
    for time_slot_id, val in speaker.time_order.items():  # new_time_order_info.items():
        time_slot = etree.SubElement(new_time_order, "TIME_SLOT")
        time_slot.set("TIME_SLOT_ID", time_slot_id)
        time_slot.set("TIME_VALUE", str(val))
    speaker.annotation_doc.replace(speaker.annotation_doc.find("TIME_ORDER"), new_time_order)

    if speaker.tiers_id != None:
        tmp = []
        last_tier_pos = speaker.annotation_doc.index(speaker.annotation_doc.findall("TIER")[-1])
        for pos, branch in enumerate(speaker.annotation_doc.getchildren()):
            if pos > last_tier_pos:
                tmp.append(branch)
                speaker.annotation_doc.remove(branch)
    else:
        tmp = []
        time_order_pos = speaker.annotation_doc.index(speaker.annotation_doc.find("TIME_ORDER"))
        for pos, branch in enumerate(speaker.annotation_doc.getchildren()):
            if pos > time_order_pos:
                tmp.append(branch)
                speaker.annotation_doc.remove(branch)

    tier = etree.SubElement(speaker.annotation_doc, "TIER")
    tier.set("LINGUISTIC_TYPE_REF", "default-lt")
    tier.set("TIER_ID", tier_id)
    for annotation_info in new_annotations_info:
        annotation = etree.SubElement(tier, "ANNOTATION")
        slot = etree.SubElement(annotation, "ALIGNABLE_ANNOTATION")
        slot.set("ANNOTATION_ID", annotation_info[0])
        slot.set("TIME_SLOT_REF1", annotation_info[1])
        slot.set("TIME_SLOT_REF2", annotation_info[2])
        value = etree.SubElement(slot, "ANNOTATION_VALUE")
        if values != None:
            value.text = annotation_info[3]
        else:
            value.text = ""

    if speaker.tiers_id == None:
        ling_type = etree.Element("LINGUISTIC_TYPE")
        ling_type.set("GRAPHIC_REFERENCES", "false")
        ling_type.set("LINGUISTIC_TYPE_ID", "default-lt")
        ling_type.set("TIME_ALIGNABLE", "true")
        speaker.annotation_doc.append(ling_type)
    for branch in tmp:
        speaker.annotation_doc.append(branch)

    if newfile:
        file = speaker.eaf_file.split('.')[0] + '_' + tier_id + '.eaf'
    else:
        file = speaker.eaf_file
    open(file, 'w').close()
    etree.indent(speaker.annotation_doc, space="    ")
    annotation_doc = etree.ElementTree(speaker.annotation_doc)
    annotation_doc.write(file, pretty_print=True, xml_declaration=True, encoding='utf-8')


def modify_tier_annotations(tier, new_anno_info):
    for a, new_anno in enumerate(new_anno_info):

        if a < len(tier.findall("ANNOTATION")):
            new_annotation = etree.Element("ANNOTATION")
            slot = etree.SubElement(new_annotation, "ALIGNABLE_ANNOTATION")
            old_annotation = tier.getchildren()[a].find("ALIGNABLE_ANNOTATION")
            slot.set("ANNOTATION_ID", old_annotation.get("ANNOTATION_ID"))
            slot.set("TIME_SLOT_REF1", new_anno_info[a][0])
            slot.set("TIME_SLOT_REF2", new_anno_info[a][1])
            value = etree.SubElement(slot, "ANNOTATION_VALUE")
            if len(new_anno_info) > 3:
                value.text = new_anno_info[a][2]
            else:
                value.text = old_annotation.getchildren()[0].text
            tier.replace(old_annotation.getparent(), new_annotation)

        else:
            new_annotation = etree.SubElement(tier, "ANNOTATION")
            slot = etree.SubElement(new_annotation, "ALIGNABLE_ANNOTATION")
            last_anno_id = tier.getparent().find("HEADER").findall("PROPERTY")[-1]
            new_anno_id = str(int(last_anno_id.text) + 1)
            slot.set("ANNOTATION_ID", new_anno_id)
            slot.set("TIME_SLOT_REF1", new_anno_info[a][0])
            slot.set("TIME_SLOT_REF2", new_anno_info[a][1])
            value = etree.SubElement(slot, "ANNOTATION_VALUE")
            if len(new_anno_info) > 3:
                value.text = new_anno_info[a][2]

            new_last_anno_id = etree.Element("PROPERTY")
            new_last_anno_id.set("NAME", "lastUsedAnnotationId")
            new_last_anno_id.text = new_anno_id
            tier.getparent().find("HEADER").replace(last_anno_id, new_last_anno_id)


def define_controlled_vocab(eaf_file, tier_id, values):
    speaker = Speaker(eaf_file)

    tmp = []
    ling_type_pos = speaker.annotation_doc.index(speaker.annotation_doc.findall("LINGUISTIC_TYPE")[-1])
    for pos, branch in enumerate(speaker.annotation_doc.getchildren()):
        if pos > ling_type_pos:
            tmp.append(branch)
            speaker.annotation_doc.remove(branch)
    ling_type = etree.Element("LINGUISTIC_TYPE")
    ling_type.set("CONTROLLED_VOCABULARY_REF", tier_id.lower())
    ling_type.set("GRAPHIC_REFERENCES", "false")
    ling_type.set("LINGUISTIC_TYPE_ID", tier_id)
    ling_type.set("TIME_ALIGNABLE", "true")
    speaker.annotation_doc.append(ling_type)
    lang = etree.Element("LANGUAGE")
    lang.set("LANG_DEF", "http://cdb.iso.org/lg/CDB-00130975-001")
    lang.set("LANG_ID", "und")
    lang.set("LANG_LABEL", "undetermined (und)")
    speaker.annotation_doc.append(lang)
    for branch in tmp:
        speaker.annotation_doc.append(branch)

    vocab = etree.Element("CONTROLLED_VOCABULARY")
    vocab.set("CV_ID", tier_id)
    description = etree.SubElement(vocab, "DESCRIPTION")
    description.set("LANG_REF", "und")
    for val in values:
        cv_entry = etree.SubElement(vocab, "CV_ENTRY_ML")
        cveid = "cveid_"
        cveid += _generate_identifier()
        cv_entry.set("CVE_ID", cveid)
        cve_val = etree.SubElement(cv_entry, "CVE_VALUE")
        cve_val.set("LANG_REF", "und")
        cve_val.text = val
    speaker.annotation_doc.append(vocab)

    open(eaf_file, 'w').close()
    etree.indent(speaker.annotation_doc, space="    ")
    annotation_doc = etree.ElementTree(speaker.annotation_doc)
    annotation_doc.write(eaf_file, pretty_print=True, xml_declaration=True, encoding='utf-8')


def _format_annotations_info(speaker, time_slots, values=None):
    slots_id = speaker._update_time_order(time_slots)
    annotations_info = []

    for i in range(0, len(slots_id), 2):
        if speaker.annotation_index != None:
            annotation_id = "a" + str(int(speaker.annotation_index[-1][1:]) + i//2 + 1)
        else:
            annotation_id = "a" + str(i//2 + 1)
        if values != None:
            annotations_info.append((annotation_id, slots_id[i], slots_id[i+1], values[i]))
        else:
            annotations_info.append((annotation_id, slots_id[i], slots_id[i+1]))

    return annotations_info


def create_fixed_slots(*eaf_files, duration=90000, mode=None):
    conv = Conversation(*eaf_files)

    if mode == None:
        slots = np.array([[t * duration, (t+1) * duration] for t in range(conv.conv_length // duration)])

    elif mode == "align_ipus":
        slots = np.array([[t * duration, (t+1) * duration] for t in range(conv.onv_length // duration)])
        silences_midpoints = conv.get_silences_midpoints()

        for s, slot in enumerate(slots):
            lowerbound = silences_midpoints[np.argmin(abs(silences_midpoints - slot[0]))]
            upperbound = silences_midpoints[np.argmin(abs(silences_midpoints - slot[1]))]
            slots[s] = [lowerbound, upperbound]

        slots[0][0] = 0
        for i in range(1, len(slots)):
            slots[i][0] += 1

    return slots


def _generate_identifier():
    identifier = ""
    for _ in range(31):
        if random.choice(['letter', 'number']) == 'letter':
            identifier += random.choice(string.ascii_letters[:26])
        else:
            identifier += str(random.randint(0, 9))
    identifier = [identifier[:8], identifier[8:12], identifier[12:16], identifier[16:20], identifier[20:]]
    identifier = '-'.join(identifier)
    return identifier


class Speaker:
    def __init__(self, eaf_file):
        self.eaf_file = eaf_file
        self.annotation_doc = etree.parse(eaf_file).getroot()
        self.time_order = self._get_time_order()
        self.conv_length = int(list(self.time_order.values())[-1]) if len(self.time_order) > 0 else None
        self.tiers_id = self._get_tiers_id()
        self.annotation_index = self._get_annotation_index()

    def _get_time_order(self):
        if len(self.annotation_doc.xpath('//TIME_SLOT')) > 0:
            time_order = {}
            for slot in self.annotation_doc.xpath('//TIME_SLOT'):
                time_order[slot.get("TIME_SLOT_ID")] = int(slot.get("TIME_VALUE"))
            return time_order
        else:
            return {}

    def _get_tiers_id(self):
        if len(self.annotation_doc.xpath('//TIER')) > 0:
            tiers_id = []
            for tier in self.annotation_doc.xpath('//TIER'):
                tiers_id.append(tier.get("TIER_ID"))
            return tiers_id
        else:
            return None

    def _get_annotation_index(self):
        if len(self.annotation_doc.xpath('//ALIGNABLE_ANNOTATION')) > 0:
            annotation_index = []
            for annotation in self.annotation_doc.xpath('//ALIGNABLE_ANNOTATION'):
                annotation_index.append(annotation.get("ANNOTATION_ID"))
            return annotation_index
        else:
            return None


    def get_annotations(self, tier_id="all"):
        if tier_id != "all":
            annotations = []
            tier = self.annotation_doc.findall("TIER")[self.tiers_id.index(tier_id)]
            tier = etree.ElementTree(tier)
            for annotation in tier.xpath('//ALIGNABLE_ANNOTATION'):
                lowerbound = self.time_order[annotation.get("TIME_SLOT_REF1")]
                upperbound = self.time_order[annotation.get("TIME_SLOT_REF2")]
                annotations.append(([lowerbound, upperbound], annotation.getchildren()[0].text))

        else:
            annotations = {}
            for annotation in self.annotation_doc.xpath('//ALIGNABLE_ANNOTATION'):
                tier_id = annotation.getparent().getparent().get("TIER_ID")
                if tier_id not in annotations.keys():
                    annotations[tier_id] = []
                lowerbound = self.time_order[annotation.get("TIME_SLOT_REF1")]
                upperbound = self.time_order[annotation.get("TIME_SLOT_REF2")]
                annotations[tier_id].append(([lowerbound, upperbound], annotation.getchildren()[0].text))

        return annotations


    def get_speech_activity(self, time_slots=True, binary=True):
        tier_id = [name for name in self.tiers_id if np.logical_or(('ipu' in name.lower()),
                                                                   ('transcription' in name.lower()))][0]
        speech_activ = self.get_annotations(tier_id)

        if time_slots:
            if binary:
                speech_activity = [(slot, 0) if np.logical_or(('#' in str(val)), (val is None)) else (slot, 1)
                                   for slot, val in speech_activ]
            else:
                speech_activity = [(slot, val) for slot, val in speech_activ]

        else:
            speech_activity = np.zeros(self.conv_length, dtype=int)
            for slot, val in speech_activ:
                if not np.logical_or(('#' in str(val)), (val is None)):
                    speech_activity[slot[0]:slot[1]] = 1

        return speech_activity


    def get_pauses(self):
        tier_id = [name for name in self.tiers_id if ('ipu' or 'transcription') in name.lower()][0]
        speech_activity = self.get_annotations(tier_id)
        pauses = [slot for slot, val in speech_activity if np.logical_or(('#' in str(val)), (val is None))]
        return pauses


    def get_ipus(self):
        tier_id = [name for name in self.tiers_id if ('ipu' or 'transcription') in name.lower()][0]
        speech_activity = self.get_annotations(tier_id)
        ipus = [slot for slot, val in speech_activity if not np.logical_or(('#' in str(val)), (val is None))]
        return ipus


    def get_feedbacks(self, verbal_only=False, slots_only=False, type=False):
        tier_id = [name for name in self.tiers_id if 'type de feedback' in name.lower()][0]
        fdbacks = self.get_annotations(tier_id)

        if verbal_only:
            feedbacks = np.zeros(self.conv_length, dtype=int)
            for slot, _ in fdbacks:
                feedbacks[slot[0]:slot[1]] = 1
            speech_activity = self.get_speech_activity(time_slots=False, binary=True)
            feedbacks[speech_activity == 0] = 0

            # align feedbacks with ipus
            feedbacks = self._align_feedbacks_with_ipus(feedbacks, speech_activity, slots_only=slots_only)

        else:
            if slots_only:
                feedbacks = np.array([slot for slot, _ in fdbacks])
            elif type:
                feedbacks = np.empty(self.conv_length, dtype='<U1')
                for slot, val in fdbacks:
                    feedbacks[np.arange(slot[0], slot[1])] = val
            else:
                feedbacks = np.zeros(self.conv_length, dtype=int)
                for slot, _ in fdbacks:
                    feedbacks[slot[0]:slot[1]] = 1

        return feedbacks


    def _align_feedbacks_with_ipus(self, feedbacks, speech_activity_ts, slots_only):
        fdbacks = [len(list(seq[1])) for seq in groupby(feedbacks)]
        fdbacks = [list(slot) for s, slot in enumerate(pairwise(np.cumsum(fdbacks)))
                   if s % 2 == 0]
        sa = self.get_speech_activity()
        sa = np.array([s[0] for s in sa])
        min_length = 100

        while len([(sa[np.where(sa == fb)[0]], fb)
                   for fb in fdbacks if len(np.where(sa == fb)[0]) == 1]) > 0:

            fdbacks = [list(seq[1]) for seq in groupby(feedbacks)]
            for i in range(len(fdbacks)):
                if len(fdbacks[i]) < min_length:
                    fdbacks[i] = [0] * len(fdbacks[i])

            fdbacks = [fb for fb in chain.from_iterable(fdbacks)]
            for i in range(1, len(fdbacks)-1):
                if fdbacks[i] == 0 and fdbacks[i-1] == 1:
                    if speech_activity_ts[i] == 1:
                        fdbacks[i] = 1

            for i in range(len(fdbacks)-2, -1, -1):
                if fdbacks[i] == 0 and fdbacks[i+1] == 1:
                    if speech_activity_ts[i] == 1:
                        fdbacks[i] = 1

            fdbacks = [len(list(seq[1])) for seq in groupby(fdbacks)]
            fdbacks = [list(slot) for s, slot in enumerate(pairwise(np.cumsum(fdbacks)))
                       if s % 2 == 0]
            min_length += 50

        if slots_only:
            return np.array(fdbacks)
        else:
            feedbacks = np.zeros(self.conv_length, dtype=int)
            for slot in fdbacks:
                feedbacks[slot[0]:slot[1]] = 1
            return feedbacks


    def _update_time_order(self, new_slots):
        new_slots_id = []
        new_slots = new_slots.flatten()

        if len(self.time_order) == 0:
            new_slots = np.sort(new_slots)
            for i, slot in enumerate(new_slots):
                new_slots_id.append("ts" + str(i))
                self.time_order["ts" + str(i)] = slot

        else:
            new_time_order = {}

            prevshift = 0
            for ts, val in self.time_order.items():
                shift = len(np.where(new_slots < val)[0])
                if shift != prevshift:
                    for i in range(shift-prevshift):
                        new_slots_id.append("ts" + str(int(ts[2:]) + prevshift + i))
                        new_time_order[new_slots_id[-1]] = new_slots[prevshift + i]
                new_time_order["ts" + str(int(ts[2:]) + shift)] = val
                prevshift = shift

            for anno_info in self.annotation_doc.xpath('//ALIGNABLE_ANNOTATION'):
                new_anno_info = etree.Element(anno_info.tag)
                new_anno_info.set(*anno_info.items()[0])
                shift1 = sum([1 if slot_val < self.time_order[anno_info.get("TIME_SLOT_REF1")] else 0
                              for slot_val in new_slots])
                shift2 = sum([1 if slot_val < self.time_order[anno_info.get("TIME_SLOT_REF2")] else 0
                              for slot_val in new_slots])
                new_anno_info.set("TIME_SLOT_REF1", "ts" + str(int(anno_info.get("TIME_SLOT_REF1")[2:]) + shift1))
                new_anno_info.set("TIME_SLOT_REF2", "ts" + str(int(anno_info.get("TIME_SLOT_REF2")[2:]) + shift2))
                new_anno_info_val = etree.SubElement(new_anno_info, anno_info.getchildren()[0].tag)
                new_anno_info_val.text = anno_info.getchildren()[0].text
                anno_info.getparent().replace(anno_info, new_anno_info)

            self.time_order = new_time_order

        return new_slots_id



class Conversation:
    def __init__(self, *eaf_files):
        self.speakers = [Speaker(eaf_file) for eaf_file in eaf_files]
        self.conv_length = self.speakers[0].conv_length


    def get_speech(self, slots_only=True, blocks=False):
        speech = np.zeros(self.conv_length, dtype=int)
        for speaker in self.speakers:
            speech_activity = speaker.get_speech_activity(time_slots=False)
            speech[speech == 0] = speech_activity[speech == 0]
        if slots_only:
            """speech = [len(list(seq[1])) for seq in groupby(speech)]
            speech = np.insert(np.cumsum(speech), 0, 0)
            speech = [[speech[i], speech[i+1]-1] for i in range(len(speech)-1)] #-1 ?"""
            breaks = np.where(np.where(speech == 1)[0] - np.roll(np.where(speech == 1)[0], 1) != 1)[0]
            speech = [[i, j] for i, j in zip(breaks[:-1], np.roll(breaks, 1)[:-1])]
        elif blocks:
            speech = [list(seq[1]) for seq in groupby(speech)]
        return speech


    def get_silences(self, slots_only=True, blocks=False):
        silences = np.ones(self.conv_length, dtype=int)
        for speaker in self.speakers:
            speech_activity = speaker.get_speech_activity()
            silences[silences == 1] -= speech_activity[silences == 1]
        if slots_only:
            """silences = [len(list(seq[1])) for seq in groupby(silences)]
            silences = np.insert(np.cumsum(silences), 0, 0)
            silences = [[silences[i], silences[i+1]-1] for i in range(len(silences)-1)]"""
            breaks = np.where(np.where(silences == 1)[0] - np.roll(np.where(silences == 1)[0], 1) != 1)[0]
            silences = [[i, j] for i, j in zip(breaks[:-1], np.roll(breaks, 1)[:-1])]
        elif blocks:
            silences = [list(seq[1]) for seq in groupby(silences)]
        return silences


    def get_silences_length(self):
        silences = self.get_silences(slots_only=False, blocks=True)
        silences_length = [len(seq) for seq in silences]
        return silences_length


    def get_silences_midpoints(self, minimal_length=None):
        speech = self.get_speech(blocks=True)
        silences_midpoints = [sum([len(s) for s in speech[:i]]) + int(len(speech[i]) / 2)
                              for i in range(len(speech)) if 0 in speech[i]]
        if minimal_length != None:
            silences_length = self.get_silences_length()
            silences_midpoints = [sm for i, sm in enumerate(silences_midpoints)
                                  if silences_length[i] > minimal_length]
        return np.array(silences_midpoints)


    def get_overlaps(self, slots_only=True, remove_feedback=True, blocks=False):
        overlaps = np.zeros(self.conv_length, dtype=int)
        for speaker in self.speakers:
            speech_activity = speaker.get_speech_activity(time_slots=False, binary=True)
            if remove_feedback:
                feedbacks = speaker.get_feedbacks(verbal_only=True, slots_only=False)
                speech_activity -= feedbacks
            overlaps += speech_activity
        overlaps[np.where(overlaps <= 1)] = 0
        overlaps[np.where(overlaps > 1)] = 1

        if slots_only:
            breaks = np.where(np.where(overlaps == 1)[0] - np.roll(np.where(overlaps == 1)[0], 1) != 1)[0]
            overlaps = [[i, j] for i, j in zip(breaks[:-1], np.roll(breaks, 1)[:-1])]
        elif blocks:
            overlaps = [list(seq[1]) for seq in groupby(overlaps)]

        return overlaps


    def get_turns(self):
        speech_info = [[speaker.get_speech_activity(binary=True),
                        speaker.get_feedbacks(verbal_only=True, slots_only=True)] for speaker in self.speakers]
        turn_taking = ([], [])

        for i in range(len(self.speakers)):
            turn = []
            for slot, val in speech_info[i][0]:

                if val == 1:
                    if np.all(speech_info[i][1] != slot):
                        if len(turn) == 0:
                            turn = slot
                        else:
                            turn = [turn[0], slot[1]]

                elif val == 0 and len(turn) > 0:
                    for j in [j for j in range(len(self.speakers)) if j != i]:
                        for sl, v in speech_info[j][0]:
                            if sl[1] > slot[0] and sl[0] < slot[1] and v == 1:
                                if np.all(speech_info[j][1] != sl):
                                    if sl[0] < slot[0]:
                                        turn_taking[i].append(turn)
                                        turn = []
                                        break
                                    elif sl[0] > slot[0]:
                                        turn = [turn[0], sl[0]]
                                        turn_taking[i].append(turn)
                                        turn = []
                                        break
                    if len(turn) > 0:
                        turn = [turn[0], slot[1]]

        return [np.array(turns) for turns in turn_taking]