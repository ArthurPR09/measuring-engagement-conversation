import os
import warnings
import numpy as np
from itertools import groupby
import pyelan as pe
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa


def read_text_grid(textgrid, item_name=None):
    """
    Parse TextGrid File. Seconds are converted into milliseconds to match ELAN output.
    ------
    Parameters:
        textgrid: TextGrid file path
        item_name: string. Item to be parsed. If None the whole TextGrid is parsed
    ------
    Returns:
        List of (xmin, xmax, text) tuples, xmin and xmax defining each time interval
    """
    remove_blanks = str.maketrans({'\n': '', '\t': ''})
    with open(textgrid, 'r') as f:
        lines = f.readlines()
    lines = [line.translate(remove_blanks).replace('"', '') for line in lines]
    items_pos = [i for i, line in enumerate(lines) if 'item' in line][1:]
    items_names = [lines[i+2].split('=')[1].strip() for i in items_pos]

    read_values = lambda lines, i, j: (round(float(lines[i+6 + 4*j + k].split('=')[1]), 2) * 1000
                                       if k < 3 else lines[i+6 + 4*j + k].split('=')[1].strip()
                                       for k in range(1, 4)) # function to extract each
                                                             # (xmin, xmax, text) tuple
    data = []

    if item_name == None:
        for i in items_pos[1:]:
            d = []
            nb_intervals = int(lines[i+5].split('=')[1]) # nb of intervals within a
                                                         # particular item section
            for j in range(nb_intervals):
                d.append(read_values(lines, i, j)) # (xmin, xmax, text) tuple for each interval
            data.append(d)

    elif item_name in items_names:
        i = items_pos[items_names.index(item_name)] # item position within the TextGrid
        nb_intervals = int(lines[i+5].split('=')[1]) # nb of intervals within a
                                                     # particular item section
        for j in range(nb_intervals):
            data.append(read_values(lines, i, j)) # (xmin, xmax, text) tuple for each interval

    return data


########## FEEDBACKS ##########

def get_feedbacks(speaker, segments, verbal=False, gen_vs_spec=False, mode="frequency"):
    """
    Computes the presence, number or frequency of each feedback type within each segment
    ------
    Parameters:
        speaker: PyElan.Speaker object
        segments: 2-dimensional ndarray. Contains time intervals
        verbal: boolean. If True the purely nonverbal feedbacks are not taken into account
        gen_vs_spec: boolean. If True specific feedback types (positive expected, negative expected,
            positive unexpected, negative unexpected) are blended together
        mode: string. Quantity of interest to be computed. Possible values: occurrence, number,
            frequency
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per feedback type
    """
    try:
        tier_fb = [tier for tier_name, tier in speaker.tiers.items()
                   if 'type de feedback' in tier_name.lower()][0]
    except IndexError:
        print(speaker.tiers.keys())
        tier_name = input("Enter tier name: ") # select the correct tier name from the
                                               # tiers names list
        tier_fb = speaker.tiers[tier_name]

    feedbacks = speaker.get_feedbacks(verbal=verbal, type=True, as_time_series=True)
    feedbacks = [feedbacks[lbound:rbound] for lbound, rbound in segments]
    feedbacks = [[seq[0] for seq in groupby(feedbacks[i])] for i in range(len(segments))]

    if gen_vs_spec:
        for key, val in tier_fb["VOC"].items():
            if val != "générique":
                tier_fb["VOC"][key] = "spécifique"

    segments = segments / 1000
    if mode == "frequency":
        feedbacks = np.array([[fb.count(val) / (seg[1]-seg[0]) for val in tier_fb["VOC"].keys()]
                              for fb, seg in zip(feedbacks, segments)])
    else:
        feedbacks = np.array([[fb.count(val) for val in tier_fb["VOC"].keys()] for fb in feedbacks])
        if mode == "number":
            pass
        if mode == "occurrence":
            feedbacks[feedbacks > 1] = 1

    feedbacks = pd.DataFrame(data=feedbacks,
                             columns=["{} feedbacks {}".format(val, mode) for val in tier_fb["VOC"].values()])
    return feedbacks


########## NONVERBAL FEATURES ##########

def get_smiles(speaker, segments, mode="frequency"):
    """
    Computes the presence, number or frequency of each type of smile types (S0, S1, S2, S3, S4)
        within each segment. Only smiles that are not part of a feedback are considered.
    ------
    Parameters:
        speaker: PyElan.Speaker object
        segments: 2-dimensional ndarray. Contains time intervals.
        mode: string. Quantity of interest to be computed. Possible values: occurrence, number,
                frequency.
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per smile type
    """
    try:
        tier_smiles = [tier for tier_name, tier in speaker.tiers.items()
                       if 'sourire' in tier_name.lower()][0]
    except IndexError:
        print(speaker.tiers.keys())
        tier_name = input("Enter tier name: ") # select the correct tier name from the
                                               # tiers names list
        tier_smiles = speaker.tiers[tier_name]
    if None in tier_smiles["VOC"].values():
        tier_smiles["VOC"].pop(tier_smiles["VAL_ID"][None])
    tier_smiles_voc = np.empty((len(tier_smiles["VOC"]), 2), dtype=object)
    for key, val in tier_smiles["VOC"].items():
        level = ''.join(x for x in val if x.isdigit())
        try:
            tier_smiles_voc[int(level), 1] = 'S' + ''.join([x for x in val if x.isdigit()])
            tier_smiles_voc[int(level), 0] = key
        except ValueError:
            tier_smiles_voc[-1, 1] = 'x'
            tier_smiles_voc[-1, 0] = key
    tier_smiles["VOC"] = dict(tier_smiles_voc)

    sml = speaker.get_annotations(tier_smiles)
    try:
        feedbacks = speaker.get_feedbacks()
        which_not_fb = [all(np.logical_or(np.logical_and(feedbacks[:, 0] > rb, feedbacks[:, 1] > rb),
                                      np.logical_and(feedbacks[:, 0] < lb, feedbacks[:, 1] < lb)))
                        for lb, rb in sml[:, :2]] # smiles that are not part of a feedback
        sml = sml[which_not_fb]
    except IndexError:
        pass

    smiles = np.empty(speaker.conv_length, dtype=int)
    for lbound, rbound, val in sml:
        smiles[lbound:rbound] = val
    smiles = [smiles[lbound:rbound] for lbound, rbound in segments]

    segments = segments / 1000
    if mode == "duration":
        smiles = np.array([[sm.count(val) / (seg[1]-seg[0]) for val in tier_smiles["VOC"].keys()]
                           for sm, seg in zip(smiles, segments)])
    else:
        smiles = [[seq[0] for seq in groupby(seg)] for seg in smiles]
        if mode == "frequency":
            smiles = np.array([[sm.count(val) / (seg[1]-seg[0]) for val in tier_smiles["VOC"].keys()]
                               for sm, seg in zip(smiles, segments)])
        else:
            smiles = np.array([[sm.count(val) for val in tier_smiles["VOC"].keys()]
                               for sm in smiles])
            if mode == "number":
                pass
            elif mode == "occurrence":
                smiles[smiles > 1] = 1

    smiles = pd.DataFrame(data=smiles[:, :5],
                          columns=["Smiles ({}) {}".format(val, mode) for val in tier_smiles["VOC"].values()][:5])
    return smiles


def get_headnods(speaker, segments, mode="frequency"):
    """
    Computes the presence, number or frequency of head nods within each segment. Only headnods
        that are not part of a feedback are considered.
    ------
    Parameters:
        speaker: PyElan.Speaker object
        segments: 2-dimensional ndarray. Contains time intervals.
        mode: string. Quantity of interest to be computed. Possible values: occurrence, number,
                frequency.
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per feedback type
    """
    try:
        tier_nods = [tier for tier_name, tier in speaker.tiers.items()
                     if 'correction' in tier_name.lower()][0]
    except IndexError:
        print(speaker.tiers.keys())
        tier_name = input("Enter tier name: ")
        tier_nods = speaker.tiers[tier_name]
    if None in tier_nods["VOC"].values():
        tier_nods["VOC"].pop(tier_nods["VAL_ID"][None])
    if len(tier_nods["VOC"]) == 1:
        tier_nods["VOC"][1] = "no nod"
        tier_nods["VAL_ID"]["no nod"] = 1

    nods = speaker.get_annotations(tier_nods)
    try:
        feedbacks = speaker.get_feedbacks()
        which_not_fb = [all(np.logical_or(np.logical_and(feedbacks[:, 0] > rb, feedbacks[:, 1] > rb),
                                          np.logical_and(feedbacks[:, 0] < lb, feedbacks[:, 1] < lb)))
                        for lb, rb in nods[:, :2]] # head nods that are not part of a feedback
        nods = nods[which_not_fb]
    except IndexError:
        pass

    headnods = np.ones(speaker.conv_length, dtype=int)
    for lbound, rbound, val in nods:
        headnods[lbound:rbound] = val
    headnods = [headnods[lbound:rbound] for lbound, rbound in segments]

    segments = segments / 1000
    if mode == "duration":
        headnods = np.array([[nod.count(val) / (seg[1]-seg[0]) for val in tier_nods["VOC"].keys()]
                         for nod, seg in zip(headnods, segments)])
    else:
        headnods = [[seq[0] for seq in groupby(seg)] for seg in headnods]
        if mode == "frequency":
            headnods = np.array([[nod.count(val) / (seg[1]-seg[0]) for val in tier_nods["VOC"].keys()]
                             for nod, seg in zip(headnods, segments)])
        else:
            headnods = np.array([[nod.count(val) for val in tier_nods["VOC"].keys()] for nod in headnods])
            if mode == "number":
                pass
            elif mode == "occurrence":
                headnods[headnods > 1] = 1

    headnods = pd.DataFrame(data=headnods[:, 0], columns=["Nods {}".format(mode)])

    return headnods


def get_blinks(segments, threshold, file="//filer/AGORA/Comprendre/HMAD/PACO_CHEESE_blink_table.txt"):
    with open(file, 'r') as f:
        lines = f.readlines()
    data = np.array([line.split('\t') for line in lines], dtype='<U8')
    blink_labels = ["B", "P"]
    speakers = list(set(data[:, 5]))

    blinks_count = np.zeros((len(speakers), len(segments), len(blink_labels)), 2)
    for xmin, xmax, duration, label, speaker in data[[0, 1, 2, 3, 5]]:
        blinks_count[speakers.index(speaker),
                     np.logical_and((segments[:, 0] <= int(xmin)), (segments[:, 1] >= int(xmax))),
                     blink_labels.index(label), 0] += 1
        blinks_count[speakers.index(speaker),
                     np.logical_and((segments[:, 0] <= int(xmin)), (segments[:, 1] >= int(xmax))),
                     blink_labels.index(label), 1] += int(duration >= threshold)

    blinks_ts = []
    for speaker in speakers:
        blinks_ts.append([data[np.logical_and(data[data[:, 5] == speaker][:, 0].astype(int) >= seg[0],
                                              data[data[:, 5] == speaker][:, 1].astype(int) <= seg[1])][:, :2]
                          for seg in segments])


########## LINGUISTIC FEATURES ##########

def get_articulation_rate(textgrid, segments):
    """
    Computes the articulation rate for each segment. The articulation rate corresponds to
        the total number of syllables divided by the total speaking time in seconds.
    ------
    Parameters:
        textgrid: TextGrid file path containing syllables duration
        segments: 2-dimensional ndarray. Contains time intervals
    ------
    Returns:
        Pandas DataFrame with one line per segment
    """
    syllables = read_text_grid(textgrid, 'SyllAlign')
    syllables = np.array([[xmin, xmax] for xmin, xmax, syl in syllables if syl != ''])
    syllables_length = [(xmax - xmin) / 1000 for xmin, xmax in syllables] # syllables duration in seconds
    articulation_rate = np.zeros((len(segments), 2))
    for i, [xmin, xmax] in enumerate(syllables):
        articulation_rate[np.logical_and((segments[:, 0] <= xmin),
                                         (segments[:, 1] >= xmax)), 0] += syllables_length[i]
        articulation_rate[np.logical_and((segments[:, 0] <= xmin),
                                         (segments[:, 1] >= xmax)), 1] += 1
    articulation_rate = articulation_rate[:, 1] / articulation_rate[:, 0]
    articulation_rate = pd.DataFrame(data=articulation_rate, columns=['Articulation rate'])
    return articulation_rate


def get_nb_words(textgrid, segments):
    """
    Computes the number of word produced within each segment
    ------
    Parameters:
        textgrid: TextGrid file path containing the words and their duration
        segments: 2-dimensional ndarray. Contains time intervals
    ------
    Returns:
        1-dimensional ndarray of size len(segments)
    """
    nb_words = np.zeros(len(segments), dtype=int)
    tokens = read_text_grid(textgrid, item_name="S-token")
    for i, [xmin, xmax, word] in enumerate(tokens):
        if word != '':
            nb_words[np.logical_and((segments[:, 0] <= xmin), (xmax <= segments[:, 1]))] += 1
    return nb_words


def get_pos_freq(textgrid, segments):
    """
    Computes the frequency of each type of POS within each segment
    ------
    Parameters:
        textgrid: TextGrid file path containing the POS and their durations
        segments: 2-dimensional ndarray. Contains time intervals.
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per type of POS
    """
    pos = read_text_grid(textgrid, item_name="S-morphosyntax")
    pos_tagger_dict = {"N": "Nouns", "V": "Verbs", "R": "Adverbs", "A": "Adjectives",
                       "P": "Pronouns", "D": "Determiners", "C": "Conjunctions",
                       "I": "Interjections", "S": "Prepositions", "W": "Punctuation", "U": "Others"}
    pos_types = list(pos_tagger_dict.keys())
    pos_freq = np.zeros((len(segments), len(pos_tagger_dict)))
    for i, [xmin, xmax, type] in enumerate(pos):
        if type != '':
            pos_freq[np.where(np.logical_and((segments[:, 0] <= xmin), (segments[:, 1] >= xmax))),
                     pos_types.index(type[0])] += 1
    nb_words = get_nb_words(textgrid, segments)
    pos_freq = pos_freq / np.repeat(nb_words, len(pos_types)).reshape(len(segments), len(pos_types))
    pos_freq = pd.DataFrame(data=pos_freq,
                            columns=['{} freq'.format(tag) for tag in list(pos_tagger_dict.values())])
    return pos_freq


########## ACOUSTIC FEATURES ##########

def get_f0_data(pitch_file, segments):
    """
    Extracts f0 values from PitchTier file. The f0 time series is then segmented. Functionals
        (min, max, mean, std, median) are finally applied to each segment.
    ------
    Parameters:
        pitch_file: PitchTier file path
        segments: 2-dimensional ndarray. Contains time intervals
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per mid-level feature
    """
    with open(pitch_file, 'r') as f:
        praat_output = f.readlines()
    timepoints, f0 = [], []

    for i in range(7, len(praat_output)-1, 3):
        timept = float('.'.join(list(map(lambda x: ''.join([y for y in x if y.isdigit()]),
                                         praat_output[i].split('.')))))
        timepoints.append(timept)
        val = float('.'.join(list(map(lambda x: ''.join([y for y in x if y.isdigit()]),
                                      praat_output[i+1].split('.')))))
        f0.append(val)
    praat_output_dict = dict(zip(timepoints, f0))

    segments = segments / 1000
    print(segments)
    timepoints = [list(filter(lambda x: x >= seg[0] and x <= seg[1], timepoints)) for seg in segments]
    f0 = [list(map(lambda x: praat_output_dict[x], timepts)) for timepts in timepoints]
    functionals = np.min, np.max, np.mean, np.std, np.median
    func_names = ["min", "max", "mean", "std", "median"]
    f0 = np.array([[f(f0_seg) for f in functionals] for f0_seg in f0])
    f0 = pd.DataFrame(data=f0, columns=["f0 ({})".format(f) for f in func_names])

    return f0


def extract_features_from_audio(audio_file, segments, window, stride):
    """
    Computes acoustic features (energy, log-energy, rmse, mfccs) from wav file. Each feature is
        computed within overlapping time windows yielding low-level feature vectors. These vectors
        are then segmented. Functionals (min, max, mean, std, median) are finally applied to each segment.
    ------
    Parameters:
        speaker: PyElan.Speaker object
        audio_file: wav file path
        segments: 2-dimensional ndarray. Contains time intervals
        window: float. windows size in seconds
        stride: float. step size in seconds
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per mid-level feature
    """
    functionals = np.min, np.max, np.mean, np.std, np.median
    apply_func = lambda x: np.array([[f(seg) for seg in x] for f in functionals])
    func_names = ["min", "max", "mean", "std", "median"]

    signal, sr = librosa.load(audio_file, sr=None)
    segments_sig = segments / 1000 * sr
    signal = [signal[lbound:rbound] for lbound, rbound in segments_sig.astype(int)]

    energy = [[sum(seg[i:int(i+window*sr)] ** 2) for i in range(0, len(seg), int(stride * sr))]
              for seg in signal]
    log_energy = [np.log(seg) for seg in energy]
    rmse = [np.sqrt(seg) / (window * sr) for seg in energy]
    energy_features = np.array([apply_func(feature) for feature in [energy, log_energy, rmse]])
    energy_features = energy_features.reshape((3 * len(functionals), len(segments))).T
    energy_features_names = ["{} ({})".format(feat, f) for feat in ['Energy', 'Log-energy', 'RMSE']
                             for f in func_names]

    mfccs = [librosa.feature.mfcc(seg, sr=sr, n_mfcc=13, hop_length=int(stride * sr),
                                  n_fft=int(window * sr))[1:] for seg in signal]
    mfccs_features = np.array([[[f(mfcc) for f in functionals] for mfcc in seg] for seg in mfccs])
    mfccs_features = mfccs_features.reshape((len(segments), 12 * len(functionals)))
    mfccs_features_names = ['MFCC{} ({})'.format(i, f) for i in range(1, 13) for f in func_names]

    acoustic_features = np.concatenate([energy_features, mfccs_features], axis=1)
    features_names = energy_features_names + mfccs_features_names
    index_values = [i for i in range(len(segments))]
    acoustic_features = pd.DataFrame(data=acoustic_features, index=index_values, columns=features_names)

    return acoustic_features


def get_acoustic_features(pitch_file, audio_file, segments, window, stride):
    """
    Computes mid-level acoustic features within each segment
    ------
    Parameters:
        pitch_file: PitchTier file path
        audio_file: wav file path
        segments: 2-dimensional ndarray. Contains time intervals
        window: float. windows size in seconds
        stride: float. step size in seconds
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per mid-level feature
    """
    f0 = get_f0_data(pitch_file, segments)
    features_from_audio = extract_features_from_audio(audio_file, segments, window, stride)
    acoustic_features = pd.concat([f0, features_from_audio], axis=1)
    return acoustic_features


########## DESIGN MATRIX ##########

def design_matrix_speak(pair, participant, segments, annotations_file, pos_file, syl_file,
                        pitch_file, audio_file):
    """
    Extracts the design matrix to model engagement when the participant is speaking
    ------
    Parameters:
        pair: string. Pair ID
        participant: string. Participant ID
        segments: 2-dimensional ndarray. Contains time intervals
        annotations_file: eaf file path containing annotations of smiles and head nods
        pos_file: TextGrid file path containing the POS and their duration
        syl_file: TextGrid file path containing the syllables duration
        praat_file: PitchTier file path
        audio_file: wav file path
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per feature
    """
    speaker = pe.Speaker(annotations_file)
    smiles = get_smiles(speaker, segments, mode="frequency")
    nods = get_headnods(speaker, segments, mode="frequency")
    pos = get_pos_freq(pos_file, segments)
    conj = pos[['Conjunctions freq']]
    modifiers = pos[['Adjectives freq', 'Adverbs freq']]
    articulation_rate = get_articulation_rate(syl_file, segments)
    acoustic_features = get_acoustic_features(pitch_file, audio_file, segments, 0.05, 0.01)

    pair_id = pd.DataFrame(data=np.repeat(pair, len(segments)), columns=['Pair id'])
    speaker_id = pd.DataFrame(data=np.repeat(participant, len(segments)), columns=['Speaker id'])
    segment_dur = np.array([(seg[1] - seg[0]) / 1000 for seg in segments]) # segments duration in seconds
    segment_dur = pd.DataFrame(data=segment_dur, columns=['Segment duration (s)'])
    design_matrix = pd.concat([pair_id, speaker_id, segment_dur, smiles, nods, articulation_rate,
                               conj, modifiers, acoustic_features], axis=1)

    return design_matrix

def design_matrix_listen(pair, participant, segments, annotations_file):
    """
    Extracts the design matrix to model engagement when the participant is listening
    ------
    Parameters:
        pair: string. Pair ID
        participant: string. Participant ID
        segments: 2-dimensional ndarray. Contains time intervals
        annotations_file: eaf file containing annotations of feedbacks, smiles and head nods
    ------
    Returns:
        Pandas DataFrame with one line per segment and one column per feature
    """
    speaker = pe.Speaker(annotations_file)
    smiles = get_smiles(speaker, segments, mode="frequency")
    nods = get_headnods(speaker, segments, mode="frequency")
    pair_id = pd.DataFrame(data=np.repeat(pair, len(segments)), columns=['Pair id'])
    speaker_id = pd.DataFrame(data=np.repeat(participant, len(segments)), columns=['Speaker id'])
    segment_dur = np.array([(seg[1] - seg[0]) / 1000 for seg in segments])
    segment_dur = pd.DataFrame(data=segment_dur, columns=['Segment duration (s)'])
    try:
        feedbacks = get_feedbacks(speaker, segments)
        design_matrix = pd.concat([pair_id, speaker_id, segment_dur, smiles, nods, feedbacks], axis=1)
    except IndexError:
        print("Incomplete design matrix (listener)")
        design_matrix = pd.concat([pair_id, speaker_id, segment_dur, smiles, nods], axis=1)

    return design_matrix


path = "C:/Users/Arthur/Documents/Engagement/annotations/paco_cheese"

corpora_audio = {"CHEESE": "//filer/AGORA/Comprendre/CHEESE/Audio",
                 "PACO": "//filer/AGORA/Comprendre/PACO/Audio"}
corpora_ling = {"CHEESE": "//filer/AGORA/Comprendre/paco-cheese/cheese",
                 "PACO": "//filer/AGORA/Comprendre/paco-cheese/paco"}

pairs_cheese = {pair: "CHEESE" for pair in os.listdir("//filer/AGORA/Comprendre/PACO-CHEESE/Cheese")}
pairs_paco = {pair: "PACO" for pair in os.listdir("//filer/AGORA/Comprendre/PACO-CHEESE/PACO")}
pairs_corpus = pairs_cheese | pairs_paco # indicates the corpus each pair belongs to


for dir, subdirs, files in os.walk(path):
    if dir != path:
        if len(subdirs) == 0:
            participant = dir.split('\\')[-1] # participant ID
            pair = dir.split('\\')[-2] # pair ID
            files = [os.path.join(dir, file).replace('\\', '/') for file in files if file.endswith('.eaf')]
            engagement_file = files[0] if files[0].endswith('eng.eaf') else files[1] # file with engagement
                                                                                     # level annotations
            annotations_file = files[1] if files[1] != engagement_file else files[0] # file with annotations of
                                                                                     # feedbacks, smiles and head nods
            sppas_folder = os.path.join(corpora_ling[pairs_corpus[pair]], pair, participant, 'sppas')
            pos_file = [os.path.join(sppas_folder, file) for file in os.listdir(sppas_folder)
                        if file.endswith('-pos.TextGrid')][0] # POS TextGrid file
            syl_file = [os.path.join(sppas_folder, file) for file in os.listdir(sppas_folder)
                        if file.endswith('-syll.TextGrid')][0] # syllables TextGrid file
            audio_file = os.path.join(corpora_audio[pairs_corpus[pair]], participant,
                                      participant + '.wav') # wav file
            praat_file = os.path.join(corpora_audio[pairs_corpus[pair]], participant,
                                      participant + '.PitchTier') # PitchTier file

            speaker = pe.Speaker(engagement_file)
            try:
                engagement_speak = speaker.get_annotations(speaker.tiers['Engagement (speaker)']) # engagement level
                                                                                                  # when holding
                                                                                                  # the turn
                eng_speak_df = pd.DataFrame(data=engagement_speak[:, 2], columns=['Engagement'])
                eng_speak_df.to_csv(os.path.join(dir, 'engagement_speak.csv'), index=False, sep=";")
                DM_speak = design_matrix_speak(pair, participant, engagement_speak[:, :2], annotations_file,
                                               pos_file, syl_file, praat_file, audio_file)
                DM_speak.to_csv(os.path.join(dir, 'design_matrix_speak.csv'), index=False, sep=";")

                engagement_listen = speaker.get_annotations(speaker.tiers['Engagement (listener)']) # engagement level
                                                                                                    # when not holding
                                                                                                    # the turn
                eng_listen_df = pd.DataFrame(data=engagement_listen[:, 2], columns=['Engagement'])
                eng_listen_df.to_csv(os.path.join(dir, 'engagement_listen.csv'), index=False, sep=";")
                DM_listen = design_matrix_listen(pair, participant, engagement_listen[:, :2], annotations_file)
                DM_listen.to_csv(os.path.join(dir, 'design_matrix_listen.csv'), index=False, sep=";")

                print(participant + ": done")

            except (IndexError, KeyError):
                pass