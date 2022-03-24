import os
from lxml import etree
import numpy as np


class PyElan:
    def __init__(self, eaf_file):
        self.eaf_file = eaf_file
        self.annotation_doc = etree.parse(eaf_file).getroot()
        self.time_order = self.get_time_order()
        self.conv_length = int(list(self.time_order.values())[-1]) #better with the media file
        self.tiers_id = self.get_tiers_id()
        self.annotation_index = self.get_annotation_index()

    def get_time_order(self):
        time_order = {}
        for slot in self.annotation_doc.xpath('//TIME_SLOT'):
            time_order[slot.get("TIME_SLOT_ID")] = int(slot.get("TIME_VALUE"))
        return time_order

    def get_tiers_id(self):
        tiers_id = []
        for tier in self.annotation_doc.xpath('//TIER'):
            tiers_id.append(tier.get("TIER_ID"))
        return tiers_id

    def get_annotation_index(self):
        annotation_index = []
        for annotation in self.annotation_doc.xpath('//ALIGNABLE_ANNOTATION'):
            annotation_index.append(annotation.get("ANNOTATION_ID"))
        return annotation_index


    def get_annotations(self, tier_id="all"):
        annotations = {}

        if tier_id != "all":
            tiers = self.annotation_doc.findall("TIER")
            tier = [t for t in tiers if t.get("TIER_ID") == tier_id][0]
            for annotation in tier.xpath('//ALIGNABLE_ANNOTATION'):
                lowerbound = str(self.time_order[annotation.get("TIME_SLOT_REF1")])
                upperbound = str(self.time_order[annotation.get("TIME_SLOT_REF2")])
                annotations[lowerbound + "-" + upperbound] = annotation.getchildren()[0].text

        else:
            for annotation in self.annotation_doc.xpath('//ALIGNABLE_ANNOTATION'):
                tier_id = annotation.getparent().getparent().get("TIER_ID")
                if tier_id not in annotations.keys():
                    annotations[tier_id] = {}
                lowerbound = str(self.time_order[annotation.get("TIME_SLOT_REF1")])
                upperbound = str(self.time_order[annotation.get("TIME_SLOT_REF2")])
                annotations[tier_id][lowerbound + "-" + upperbound] = annotation.getchildren()[0].text

        return annotations


    def get_pauses(self):
        pauses = []
        for val in self.annotation_doc.xpath('//ANNOTATION_VALUE'):
            if val.text == "#":
                lowerbound = self.time_order[val.getparent().get("TIME_SLOT_REF1")]
                upperbound = self.time_order[val.getparent().get("TIME_SLOT_REF2")]
                pauses.append([lowerbound, upperbound])
        return pauses


    def get_speech_activity(self, *speakers):
        conv_length = speakers[0].conv_length
        speech_activity = np.zeros(conv_length, dtype=int)
        for speaker in speakers:
            sa = np.ones(conv_length, dtype=int)
            pauses = speaker.get_pauses()
            for pause in pauses:
                sa[pause[0]:pause[1]] = 0
            speech_activity += sa
        speech_activity[np.where(speech_activity > 1)] = 1
        return speech_activity


    def get_speech_activity_blocks(self, *speakers):
        speech_activity = self.get_speech_activity(*speakers)
        speech_activity_blocks = []
        tmp = [1]
        for i in range(1, len(speech_activity)):
            if speech_activity[i] == 1:
                if speech_activity[i] == speech_activity[i-1]:
                    tmp.append(1)
                else:
                    speech_activity_blocks.append(tmp)
                    tmp = [1]
            else:
                if speech_activity[i] == speech_activity[i-1]:
                    tmp.append(0)
                else:
                    speech_activity_blocks.append(tmp)
                    tmp = [0]
        speech_activity_blocks.append(tmp)
        return speech_activity_blocks


    def get_silences_length(self, *speakers):
        speech_activity_blocks = self.get_speech_activity_blocks(*speakers)
        silences_length = [len(b) for b in speech_activity_blocks if 0 in b]
        return silences_length


    def get_silences_midpoints(self, *speakers, long=False):
        speech_activity = self.get_speech_activity_blocks(*speakers)
        silences_midpoints = [sum([len(s) for s in speech_activity[:i]]) + int(len(speech_activity[i])/2)
                             for i in range(len(speech_activity)) if 0 in speech_activity[i]]
        if long:
            silences_length = self.get_silences_length(*speakers)
            silences_midpoints = [sm for i, sm in enumerate(silences_midpoints)
                                  if silences_length[i] > 500]
        return np.array(silences_midpoints)


    def update_time_order(self, new_slots):
        new_time_order = {}
        new_slots = new_slots.flatten()
        new_slots_id = []

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


    def format_annotations_info(self, time_slots, values=None):
        slots_id = self.update_time_order(time_slots)
        annotations_info = []
        for i in range(0, len(slots_id), 2):
            annotation_id = "a" + str(int(self.annotation_index[-1][1:]) + int(i/2) + 1)
            if values != None:
                annotations_info.append((annotation_id, slots_id[i], slots_id[i+1], values[i]))
            else:
                annotations_info.append((annotation_id, slots_id[i], slots_id[i+1]))
        return annotations_info


    def modify_tier_annotations(self, tier, new_anno_info):
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


    def create_elan_tier(self, tier_id, time_slots, values=None, writefile=False):
        new_annotations_info = self.format_annotations_info(time_slots)

        last_anno_id = self.annotation_doc.find("HEADER").findall("PROPERTY")[-1]
        new_last_anno_id = etree.Element("PROPERTY")
        new_last_anno_id.set("NAME", "lastUsedAnnotationId")
        new_last_anno_id.text = str(int(last_anno_id.text) + len(time_slots.flatten()))
        self.annotation_doc.find("HEADER").replace(last_anno_id, new_last_anno_id)

        new_time_order = etree.Element("TIME_ORDER")
        for time_slot_id, val in self.time_order.items(): #new_time_order_info.items():
            time_slot = etree.SubElement(new_time_order, "TIME_SLOT")
            time_slot.set("TIME_SLOT_ID", time_slot_id)
            time_slot.set("TIME_VALUE", str(val))
        self.annotation_doc.replace(self.annotation_doc.find("TIME_ORDER"), new_time_order)

        tmp = []
        last_tier_pos = self.annotation_doc.index(self.annotation_doc.findall("TIER")[-1])
        for pos, branch in enumerate(self.annotation_doc.getchildren()):
            if pos > last_tier_pos:
                tmp.append(branch)
                self.annotation_doc.remove(branch)

        tier = etree.SubElement(self.annotation_doc, "TIER")
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
                
        for branch in tmp:
            self.annotation_doc.append(branch)

        if writefile:
            new_file = self.eaf_file
        else:
            new_file = self.eaf_file.split('.')[0] + '_' + tier_id + '.eaf'
        open(new_file, 'w').close()
        etree.indent(self.annotation_doc, space="    ")
        new_annotation_doc = etree.ElementTree(self.annotation_doc)
        new_annotation_doc.write(new_file, pretty_print=True, xml_declaration=True, encoding='utf-8')


    def create_fixed_slots(self, *speakers, duration=90000, mode=None, writefile=False):
        if mode == None:
            slots = np.array([[t * duration, (t+1) * duration] for t in range(self.conv_length // duration)])

        elif mode == "align_ipus":
            slots = np.array([[t * duration, (t+1) * duration] for t in range(self.conv_length // duration)])
            silences_midpoints = self.get_silences_midpoints(*speakers)

            for s, slot in enumerate(slots):
                lowerbound = silences_midpoints[np.argmin(abs(silences_midpoints - slot[0]))]
                upperbound = silences_midpoints[np.argmin(abs(silences_midpoints - slot[1]))]
                slots[s] = [lowerbound, upperbound]

            slots[0][0] = 0
            for i in range(1, len(slots)):
                slots[i][0] += 1

        if writefile:
            print("...")
        else:
            return slots
