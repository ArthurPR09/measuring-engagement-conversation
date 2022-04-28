from lxml import etree
import numpy as np
from itertools import groupby, chain, pairwise
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
    #descriptor.set("MEDIA_URL", "file:///" + media_file)
    descriptor.set("MEDIA_URL", media_file)
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


def define_controlled_vocab(eaf_file, vocab_id, values):
    speaker = Speaker(eaf_file)

    vocab = etree.Element("CONTROLLED_VOCABULARY")
    vocab.set("CV_ID", vocab_id)
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

    _write_eaf_file(eaf_file, speaker.annotation_doc)


def define_ling_type(eaf_file, ling_type_id, vocab_id):
    speaker = Speaker(eaf_file)

    tmp = []
    if speaker.annotation_doc.find("LINGUISTIC_TYPE") != None:
        i = speaker.annotation_doc.index(speaker.annotation_doc.findall("LINGUISTIC_TYPE")[-1])
    elif speaker.annotation_doc.find("TIER") != None:
        i = speaker.annotation_doc.index(speaker.annotation_doc.findall("TIER")[-1])
    else:
        i = speaker.annotation_doc.index(speaker.annotation_doc.find("TIME_ORDER"))
    for j, branch in enumerate(speaker.annotation_doc.getchildren()):
        if j > i:
            tmp.append(branch)
            speaker.annotation_doc.remove(branch)

    ling_type = etree.Element("LINGUISTIC_TYPE")
    ling_type.set("CONTROLLED_VOCABULARY_REF", vocab_id)
    ling_type.set("GRAPHIC_REFERENCES", "false")
    ling_type.set("LINGUISTIC_TYPE_ID", ling_type_id)
    ling_type.set("TIME_ALIGNABLE", "true")
    speaker.annotation_doc.append(ling_type)

    if speaker.annotation_doc.find("LANGUAGE") == None:
        lang = etree.Element("LANGUAGE")
        lang.set("LANG_DEF", "http://cdb.iso.org/lg/CDB-00130975-001")
        lang.set("LANG_ID", "und")
        lang.set("LANG_LABEL", "undetermined (und)")
        speaker.annotation_doc.append(lang)

    for branch in tmp:
        speaker.annotation_doc.append(branch)

    _write_eaf_file(eaf_file, speaker.annotation_doc)


def add_tier(eaf_file, tier_id, time_slots, values=None, ling_type="default-lt", newfile=False):
    speaker = Speaker(eaf_file)
    new_annotations_info = _format_annotations_info(speaker, time_slots, values=values)

    last_anno_id = speaker.annotation_doc.find("HEADER").findall("PROPERTY")[-1]
    new_last_anno_id = etree.Element("PROPERTY")
    new_last_anno_id.set("NAME", "lastUsedAnnotationId")
    new_last_anno_id.text = str(int(last_anno_id.text) + len(time_slots.flatten()//2))
    speaker.annotation_doc.find("HEADER").replace(last_anno_id, new_last_anno_id)

    new_time_order = etree.Element("TIME_ORDER")
    for time_slot_id, val in speaker.time_order.items():
        time_slot = etree.SubElement(new_time_order, "TIME_SLOT")
        time_slot.set("TIME_SLOT_ID", time_slot_id)
        time_slot.set("TIME_VALUE", str(val))
    speaker.annotation_doc.replace(speaker.annotation_doc.find("TIME_ORDER"), new_time_order)

    tmp = []
    if len(speaker.tiers) > 0:
        i = speaker.annotation_doc.index(speaker.annotation_doc.findall("TIER")[-1])
    else:
        i = speaker.annotation_doc.index(speaker.annotation_doc.find("TIME_ORDER"))
    for j, branch in enumerate(speaker.annotation_doc.getchildren()):
        if j > i:
            tmp.append(branch)
            speaker.annotation_doc.remove(branch)

    tier = etree.SubElement(speaker.annotation_doc, "TIER")
    tier.set("LINGUISTIC_TYPE_REF", ling_type)
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

    if len(speaker.tiers) > 0:
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
    _write_eaf_file(file, speaker.annotation_doc)


"""def rename_tier(eaf_file, old_id, new_id):
    speaker = Speaker(eaf_file)
    try:
        tier = speaker.annotation_doc.findall("TIER")[list(speaker.tiers.keys()).index(old_id)]"""

def modify_tier_ling_type(eaf_file, tier_id, ling_type_id):
    speaker = Speaker(eaf_file)
    if tier_id in speaker.tiers.keys():
        tier = speaker.annotation_doc.findall("TIER")[list(speaker.tiers.keys()).index(tier_id)]
        tier.set("LINGUISTIC_TYPE_REF", ling_type_id)
    _write_eaf_file(eaf_file, speaker.annotation_doc)


def merge_files(eaf_file1, eaf_file2):
    speaker = Speaker(eaf_file2)
    for tier in speaker.tiers.values():
        annotations = speaker.get_annotations(tier)
        time_slots = annotations[:, :2]
        values = annotations[:, 2]
        add_tier(eaf_file1, time_slots=time_slots, values=values, ling_type=tier["LT_REF"])


def _update_time_order(speaker, new_slots):
    new_ts_id = []
    new_slots = new_slots.flatten()

    if len(speaker.time_order) == 0:
        time_values = np.sort(new_slots)
        for i, val in enumerate(time_values):
            new_ts_id.append("ts" + str(i))
            speaker.time_order["ts" + str(i)] = val

    else:
        new_time_order = {}
        old_ts_id = {}
        time_values = [[ots, 0] for ots in speaker.time_order.values()] + [[nts, 1] for nts in new_slots]
        time_values = np.array(time_values)
        time_values = time_values[np.argsort(time_values[:, 0])]
        for i, [val, new] in enumerate(time_values):
            new_time_order["ts" + str(i)] = val
            if bool(new):
                new_ts_id.append("ts" + str(i))
            else:
                old_ts_id["ts" + str(i)] = val
        if any([n == o for n, o in zip(new_ts_id, list(old_ts_id.keys()))]):
            print("Error")
        _update_anno_ts_ref(speaker, old_ts_id)
        speaker.time_order = new_time_order

    return new_ts_id


def _update_anno_ts_ref(speaker, new_time_order):
    for anno in speaker.annotation_doc.xpath('//ALIGNABLE_ANNOTATION'):
        new_anno = etree.Element(anno.tag)
        new_anno.set(*anno.items()[0])
        old_ts_ref = [anno.get("TIME_SLOT_REF1"), anno.get("TIME_SLOT_REF2")]
        for key, val in new_time_order.items():
            if val == speaker.time_order[old_ts_ref[0]] and "TIME_SLOT_REF1" not in new_anno.attrib.keys():
                new_anno.set("TIME_SLOT_REF1", key)
            elif val == speaker.time_order[old_ts_ref[1]] and "TIME_SLOT_REF2" not in new_anno.attrib.keys():
                new_anno.set("TIME_SLOT_REF2", key)
                break
        del new_time_order[new_anno.get("TIME_SLOT_REF1")]
        del new_time_order[new_anno.get("TIME_SLOT_REF2")]
        new_anno_info_val = etree.SubElement(new_anno, anno.getchildren()[0].tag)
        new_anno_info_val.text = anno.getchildren()[0].text
        anno.getparent().replace(anno, new_anno)


def _format_annotations_info(speaker, time_slots, values=None):
    slots_id = _update_time_order(speaker, time_slots)
    annotations_info = []

    for i in range(0, len(slots_id), 2):
        if len(speaker.annotation_index) > 0:
            annotation_id = "a" + str(int(speaker.annotation_index[-1][1:]) + i//2 + 1)
        else:
            annotation_id = "a" + str(i//2 + 1)
        if values != None:
            annotations_info.append((annotation_id, slots_id[i], slots_id[i+1], values[i]))
        else:
            annotations_info.append((annotation_id, slots_id[i], slots_id[i+1]))

    return annotations_info


def _write_eaf_file(eaf_file, annotation_doc):
    open(eaf_file, 'w').close()
    etree.indent(annotation_doc, space="    ")
    annotation_doc = etree.ElementTree(annotation_doc)
    annotation_doc.write(eaf_file, pretty_print=True, xml_declaration=True, encoding='utf-8')


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
        self.ctrl_vocs = self._get_ctrl_vocs()
        self.ling_types = self._get_ling_types()
        self.tiers = self._get_tiers()
        self.annotation_index = self._get_annotation_index()
        self.speech_activity = None

    def _get_time_order(self):
        time_order = {}
        for slot in self.annotation_doc.xpath('//TIME_SLOT'):
            time_order[slot.get("TIME_SLOT_ID")] = int(slot.get("TIME_VALUE"))
        return time_order

    def _get_annotation_index(self):
        annotation_index = []
        for annotation in self.annotation_doc.xpath('//ALIGNABLE_ANNOTATION'):
            annotation_index.append(annotation.get("ANNOTATION_ID"))
        return annotation_index

    def _get_ctrl_vocs(self):
        cvs = {}
        for ctrl_voc in self.annotation_doc.xpath('//CONTROLLED_VOCABULARY'):
            cvs[ctrl_voc.get("CV_ID")] = {}
            for i, entry in enumerate(ctrl_voc.getchildren()[1:]):
                cvs[ctrl_voc.get("CV_ID")][i] = entry.getchildren()[0].text
        return cvs

    def _get_ling_types(self):
        lts = {}
        for lt in self.annotation_doc.xpath('//LINGUISTIC_TYPE'):
            lts[lt.get("LINGUISTIC_TYPE_ID")] = {"CV_ID": lt.get("CONTROLLED_VOCABULARY_REF")}
        return lts

    def _get_tiers(self):
        tiers = {}
        for tier in self.annotation_doc.findall('TIER'):
            tier_id = tier.get("TIER_ID")
            tiers[tier_id] = {"XML_OBJ": tier, "LT_REF": tier.get("LINGUISTIC_TYPE_REF")}
            try:
                tiers[tier_id]["VOC"] = self._get_tier_voc(tiers[tier_id])
                tiers[tier_id]["VAL_ID"] = self._tier_val_id(tiers[tier_id]["VOC"])
            except KeyError:
                print("No tier has been defined")
        return tiers

    def _get_tier_voc(self, tier, remove_none=True):
        tier_voc = {}

        if tier["LT_REF"] == "default-lt":
            tier = etree.ElementTree(tier["XML_OBJ"])
            i = 0
            for annotation in tier.xpath('//ALIGNABLE_ANNOTATION'):
                val = annotation.getchildren()[0].text
                val = val.strip().lower() if val != None else val
                if val not in tier_voc.values():
                    tier_voc[i] = val
                    i += 1

        else:
            try:
                tier_voc = self.ctrl_vocs[self.ling_types[tier["LT_REF"]]["CV_ID"]]
            except KeyError:
                print("No such linguistic type has been defined")

        return tier_voc

    @staticmethod
    def _tier_val_id(tier_voc):
        return dict(zip(tier_voc.values(), tier_voc.keys()))


    def get_annotations(self, tier, time_slots_only=False):
        annotations = []
        tierxml = etree.ElementTree(tier["XML_OBJ"])
        for annotation in tierxml.xpath('//ALIGNABLE_ANNOTATION'):
            leftbound = self.time_order[annotation.get("TIME_SLOT_REF1")]
            rightbound = self.time_order[annotation.get("TIME_SLOT_REF2")]
            val = annotation.getchildren()[0].text
            val = val.strip().lower() if val != None else val
            if time_slots_only:
                annotations.append([leftbound, rightbound])
            else:
                try:
                    annotations.append([leftbound, rightbound, tier["VAL_ID"][val]])
                except KeyError:
                    pass
                    #print("{} not in tier vocabulary (slot: [{}-{}])".format(val, leftbound, rightbound))
        annotations = np.array(annotations, dtype=int)
        return annotations


    def get_speech_activity(self, as_time_series=False):
        tier_ipu = [tier for tier_id, tier in self.tiers.items()
                    if np.logical_or(('ipu' in tier_id.lower()), ('transcription' in tier_id.lower()))][0]
        silence_vals = [tier_ipu["VAL_ID"][val] for val in [None, '#', '*', '@']
                        if val in tier_ipu["VAL_ID"].keys()]
        s_act = self.get_annotations(tier_ipu)
        speech_activity = np.ones(self.conv_length, dtype=int)
        for lbound, rbound, _ in s_act[np.isin(s_act[:, 2], silence_vals)]:
            speech_activity[lbound:rbound] = 0

        tier_token = [tier for tier_id, tier in self.tiers.items() if 'token' in tier_id.lower()][0]
        tokens = self.get_annotations(tier_token)
        tokens = tokens[np.logical_and((tokens[:, 2] == tier_token["VAL_ID"][None]),
                                       (tokens[:, 1] - tokens[:, 0] > 1000))]
        for lbound, rbound, _ in tokens:
            speech_activity[lbound:rbound] = 0

        if not as_time_series:
            s_act = np.array([[len(list(seq[1])), seq[0]] for seq in groupby(speech_activity)])
            speech_activity = np.cumsum(s_act[:, 0])
            speech_activity = np.insert(speech_activity, 0, 0)
            speech_activity = np.concatenate((speech_activity[:-1].reshape(-1, 1),
                                              speech_activity[1:].reshape(-1, 1),
                                              s_act[:, 1].reshape(-1, 1)), axis=1)
            if speech_activity[-1, 1] != self.conv_length:
                if speech_activity[-1, 2] == 1:
                    speech_activity = np.append(speech_activity, [speech_activity[-1, 1], self.conv_length, 0])
                else:
                    speech_activity[-1] = [speech_activity[-1, 0], self.conv_length, 0]

        self.speech_activity = speech_activity
        return speech_activity


    def get_speech_tokens(self):
        tier_token = [tier for tier_id, tier in self.tiers.items() if 'token' in tier_id.lower()][0]
        tokens = self.get_annotations(tier_token)
        return tokens


    def get_pauses(self):
        speech_activity = self.get_speech_activity(as_time_series=False)
        pauses = speech_activity[speech_activity[:, 2] == 0][:, :2]
        return pauses


    def get_ipus(self):
        speech_activity = self.get_speech_activity(as_time_series=False)
        ipus = speech_activity[speech_activity[:, 2] == 1][:, :2]
        return ipus


    def get_feedbacks(self, verbal=False, type=False, as_time_series=False):
        tier_fb = [tier for tier_id, tier in self.tiers.items() if 'type de feedback' in tier_id.lower()][0]
        fdbacks = self.get_annotations(tier_fb)

        if verbal:
            if self.speech_activity is None:
                speaking = self.get_ipus()
            else:
                speaking = self.speech_activity[self.speech_activity[:, 2] == 1][:, :2]
            fdbacks = self._align_slots(fdbacks, speaking)

        if type:
            if as_time_series:
                feedbacks = np.empty(self.conv_length, dtype=int) #dtype='<U20')
                for lbound, rbound, val in fdbacks:
                    feedbacks[lbound:rbound] = val #tier_fb["VOC"][val]
            else:
                feedbacks = fdbacks
                #feedbacks = np.array([[str(lbound), str(rbound), tier_fb["VOC"][val]]
                                      #for lbound, rbound, val in fdbacks])

        else:
            if as_time_series:
                feedbacks = np.zeros(self.conv_length, dtype=int)
                for lbound, rbound, _ in fdbacks:
                    feedbacks[lbound:rbound] = 1
            else:
                feedbacks = np.array([[lbound, rbound] for lbound, rbound, _ in fdbacks])

        return feedbacks


    def _align_slots(self, tier_a, tier_b):
        new_tier_a = []
        if tier_b.shape[1] == 3:
            tier_b = tier_b[:, 2]
        for i, (lbound, rbound, val) in enumerate(tier_a):
            if [lbound, rbound] in tier_b:
                new_tier_a.append([lbound, rbound, val])
            else:
                for slot in tier_b:
                    if self._slots_overlap([lbound, rbound], slot):
                        new_tier_a.append([slot[0], slot[1], val])
        return np.array(new_tier_a)


    @staticmethod
    def _slots_overlap(slota, slotb):
        return (slotb[0] <= slota[0] and slota[0] <= slotb[1]) or (slota[0] <= slotb[0] and slotb[0] <= slota[1])


class Conversation:
    def __init__(self, *eaf_files, speakers_id=None):
        self.speakers = [Speaker(eaf_file) for eaf_file in eaf_files]
        self.speakers_id = speakers_id
        self.conv_length = self.speakers[0].conv_length


    def get_speech(self, slots_only=True, blocks=False):
        speech = np.zeros(self.conv_length, dtype=int)
        for speaker in self.speakers:
            speech_activity = speaker.get_speech_activity(as_time_series=True)
            speech[speech == 0] = speech_activity[speech == 0]
        if slots_only:
            breaks = np.cumsum([len(list(seq[1])) for seq in groupby(speech)])
            if len(speech)-1 not in breaks:
                if breaks[-1] == len(speech):
                    breaks = breaks[:-1]
                breaks = np.insert(breaks, len(breaks), len(speech)-1)
            elif breaks[-1] == len(speech):
                breaks = breaks[:-1]
            breaks = np.array([list(slot) for slot in pairwise(breaks)])
            return breaks
        elif blocks:
            speech = [list(seq[1]) for seq in groupby(speech)]
        return speech


    """def get_silences(self, slots_only=True, blocks=False):
        silences = np.ones(self.conv_length, dtype=int)
        for speaker in self.speakers:
            speech_activity = speaker.get_speech_activity()
            silences[silences == 1] -= speech_activity[silences == 1]
        if slots_only:
        elif blocks:
            silences = [list(seq[1]) for seq in groupby(silences)]
        return silences


    def get_silences_length(self):
        silences = self.get_silences(slots_only=False, blocks=True)
        silences_length = [len(seq) for seq in silences]
        return silences_length"""


    def get_overlaps(self, slots_only=True, remove_feedback=True, blocks=False):
        overlaps = np.zeros(self.conv_length, dtype=int)
        for speaker in self.speakers:
            speech_activity = speaker.get_speech_activity(as_time_series=True)
            if remove_feedback:
                feedbacks = speaker.get_feedbacks(verbal_only=True, as_time_series=True)
                speech_activity -= feedbacks
            overlaps += speech_activity
        overlaps[np.where(overlaps <= 1)] = 0
        overlaps[np.where(overlaps > 1)] = 1

        if slots_only:
            pass
        elif blocks:
            overlaps = [list(seq[1]) for seq in groupby(overlaps)]

        return overlaps


    def get_turns(self): #remove_short=False):
        speech_info = [[speaker.get_speech_activity(as_time_series=False),
                        speaker.get_feedbacks(verbal=True, as_time_series=False)] for speaker in self.speakers]
        turn_taking = [[], []]

        for i in range(len(self.speakers)):
            turn = []

            for lbound, rbound, val in speech_info[i][0]:
                if val == 1:
                    if np.all(speech_info[i][1] != [lbound, rbound]):
                        if len(turn) == 0:
                            turn = [lbound, rbound]
                        else:
                            turn[1] = rbound

                elif val == 0 and len(turn) > 0:
                    for j in [j for j in range(len(self.speakers)) if j != i]:
                        for lb, rb, v in speech_info[j][0]:
                            if rb > lbound and lb < rbound and v == 1:
                                if np.all(speech_info[j][1] != [lb, rb]):
                                    if lb < lbound: # interlocutor's turn begins before end of
                                                    # locutor's turn
                                        turn_taking[i].append(turn)
                                        turn = []
                                        break
                                    elif lb > lbound: # interlocutor's turn begins after end of
                                                      # locutor's turn
                                        turn[1] = lb
                                        turn_taking[i].append(turn)
                                        turn = []
                                        break
                    if len(turn) > 0:
                        turn[1] = rbound

            if len(turn) > 0:
                turn_taking[i].append(turn)

        turn_taking = [np.array(turns) for turns in turn_taking]
        #if remove_short:
            #for s in range(len(turn_taking)):
                #turn_taking[s] = self._remove_short_segments(turn_taking[s])

        return turn_taking


    def _remove_short_segments(self, turns):
        segments = [t for t in chain.from_iterable(turns)]
        if segments[0] != 0:
            segments.insert(0, 0)
        if segments[-1] != self.conv_length:
            segments.insert(len(segments), self.conv_length)

        turns_ts = np.zeros(self.conv_length, dtype=int)
        for lbound, rbound in turns:
            turns_ts[lbound:rbound] = 1

        turns = turns[turns[:, 1] - turns[:, 0] > 1000]
        btw_turns = np.array([btw_turn for btw_turn in pairwise(segments) if btw_turn not in turns])
        btw_turns = btw_turns[btw_turns[:, 1] - btw_turns[:, 0] > 1000]

        gaps = np.concatenate((turns, btw_turns))
        gaps = np.sort(gaps, 0)
        gaps = np.stack((gaps[:, 1], np.roll(gaps[:, 0], -1)), axis=-1)
        gaps = gaps[gaps[:, 0] != gaps[:, 1]][:-1]
        for gap in gaps:
            if turns_ts[gap[0]-1] == 1 and turns_ts[gap[1]+1] == 1:
                turns = np.append(turns, np.array([gap]), axis=0)
            elif sum(turns_ts[gap[0]:gap[1]]) / len(turns_ts[gap[0]:gap[1]]) >= 0.5 and \
                    turns_ts[gap[0]-1] == 1 or turns_ts[gap[1]+1] == 1:
                turns = np.append(turns, np.array([gap]), axis=0)

        turns = np.sort([t for t in chain.from_iterable(turns)], axis=0)
        for bound in turns:
            if (turns == bound).sum() > 1:
                turns = turns[np.where(turns != bound)]
        turns = np.array([slot for i, slot in enumerate(pairwise(turns)) if i%2 == 0])

        return turns