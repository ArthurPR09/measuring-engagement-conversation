import os
import numpy as np
from itertools import pairwise, chain
import pyelan as pe
import shutil

input_dir = ...
output_dir = ...
anno_folder = os.path.join(input_dir, "annotations").replace('\\', '/')
anno_files = [file for file in os.listdir(anno_folder)
              if file.endswith('.eaf') and file.split('_')[3] != "eng.eaf"]

pairs_cheese = {pair: "CHEESE" for pair in os.listdir(input_dir + "/Cheese")}
pairs_paco = {pair: "PACO" for pair in os.listdir(input_dir + "/PACO")}
pairs_corpus = pairs_cheese | pairs_paco


for file1 in anno_files:
    for file2 in anno_files:
        if file1.split('_')[1] == file2.split('_')[1] and file1 != file2:
            speakers_id = [file1.split('_')[2], file2.split('_')[2]]
            pair = speakers_id[0] + '_' + speakers_id[1]
            if pair not in pairs_corpus.keys():
                file1, file2 = file2, file1
                speakers_id = speakers_id[::-1]
                pair = speakers_id[0] + '_' + speakers_id[1]
            files = tuple([os.path.join(anno_folder, file).replace('\\', '/')
                           for file in [file1, file2]])
            conv = pe.Conversation(*files)

            all_turns = conv.get_turns()
            all_btw_turns = []
            for i in range(len(conv.speakers)):
                btw_turns = [slot for slot in chain.from_iterable(all_turns[i])]
                if btw_turns[0] != 0:
                    btw_turns.insert(0, 0)
                if btw_turns[-1] != conv.conv_length:
                    btw_turns = np.append(btw_turns, conv.conv_length)
                btw_turns = np.array([slot for j, slot in enumerate(pairwise(btw_turns)) if j%2 == 0])
                btw_turns = btw_turns[btw_turns[:, 1] - btw_turns[:, 0] >= 1000]
                all_btw_turns.append(btw_turns)
            all_turns = [turns[turns[:, 1] - turns[:, 0] >= 1000] for turns in all_turns]

            video_folder = os.path.join(input_dir, pairs_corpus[pair], pair)
            try:
                m_file = [file for file in os.listdir(video_folder) if file.endswith('.mp4')][0]
                media_file = os.path.join(video_folder, m_file)
            except IndexError:
                break
            pair_folder = os.path.join(output_dir, pair)
            os.mkdir(pair_folder)
            shutil.copy(media_file, os.path.join(output_dir, pair))
            media_file = os.path.join(output_dir, m_file).replace('\\', '/')

            pair = ''.join(pair.split('_'))
            for i, speaker in enumerate(conv.speakers):
                speaker_folder = os.path.join(pair_folder, pair + "_" + speakers_id[i])
                os.mkdir(speaker_folder)
                new_file = speaker.eaf_file.split('/')[-1]
                new_file = '_'.join(new_file.split('_')[:3]) + "_eng.eaf"
                new_file = os.path.join(speaker_folder, new_file)
                shutil.copy(files[i], speaker_folder)

                pe.create_eaf_file(new_file, media_file)
                pe.define_controlled_vocab(new_file, "Engagement", [str(j) for j in range(1, 6)])
                pe.define_ling_type(new_file, "engagement", "Engagement")
                pe.add_tier(new_file, "Engagement (speaker)", all_turns[i], ling_type="engagement")
                pe.add_tier(new_file, "Engagement (listener)", all_btw_turns[i], ling_type="engagement")

            anno_files.pop(anno_files.index(file2))
            print(pair + ": done")
            break