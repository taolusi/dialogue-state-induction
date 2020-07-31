import json
import re
from functools import partial

from stanfordcorenlp import StanfordCoreNLP


class Annotator:
    def __init__(self, cfg):
        self.cfg = cfg

        # Init Stanford CoreNLP server.
        self.CORENLP_HOME = self.cfg.nlp["server"]

        self.stopwords = []
        with open(self.cfg.stopwords, 'r') as fin:
            for line in fin:
                self.stopwords.append(line.strip())
        print("Stopwords read!", end='\n\n')

    def process(self, dialog_paths):
        """ Process each dialog file and write corresponding annotation to file."""
        for dialog_path in dialog_paths:
            nlp_server = StanfordCoreNLP(self.CORENLP_HOME, memory='8g')
            self._annotator = partial(
                nlp_server.annotate,
                properties=self.cfg.nlp["props"])
            print(f"Stanford-corenlp initialized!")

            print(f"Processing {dialog_path.stem}...")

            with open(dialog_path, "r") as fin:
                dialogs = json.load(fin)
                dialog_annotations = []
                for dialog in dialogs:
                    dialog_annotations.append(self.process_dialog(dialog))

                with open(dialog_path.with_suffix(".annotation"), "w") as fout:
                    json.dump(dialog_annotations, fout, indent=2)

            nlp_server.close()
            print("Stanford-corenlp closed!", end='\n\n')

    def process_dialog(self, dialog):
        """Process each dialog."""
        dialog_annotation = dict(
            dialogue_id=dialog["dialogue_id"],
            services=dialog["services"],
            turns=[]
        )
        prev_states = {}
        for turn_idx, turn in enumerate(dialog["turns"]):
            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                user_frames = {f["service"]: f for f in turn["frames"]}
                if turn_idx > 0:
                    system_turn = dialog["turns"][turn_idx - 1]
                    system_utterance = system_turn["utterance"]
                else:
                    system_utterance = ""

                turn_annotation, prev_states = self.process_turn(
                    turn_idx, user_utterance, system_utterance, user_frames, prev_states)
                dialog_annotation["turns"].append(turn_annotation)

        return dialog_annotation

    def process_turn(
            self,
            turn_idx,
            user_utterance,
            system_utterance,
            user_frames,
            prev_states):
        """Get state update and candidate annotation for each turn.
        States are typical dialogue states, states update are those differences
        between the user goal for the current turn and preceding user turn."""
        turn_annotation = dict(
            turn_id=turn_idx,
            user_utterance=user_utterance,
            system_utterance=system_utterance)
        states = {}
        states_update = {}
        for service, frame in user_frames.items():
            states[service] = frame["state"]
            state = frame["state"]["slot_values"]
            state_update = self._get_state_update(
                state, prev_states.get(service, {}))
            states_update[service] = state_update
            states_update[service]["requested_slots"] = frame["state"]["requested_slots"]

        turn_annotation["states"] = states
        turn_annotation["states_update"] = states_update

        system_annotation, user_annotation = annotate(
            system_utterance,
            user_utterance,
            self._annotator)
        candidates, tokens = self.get_candidates(
            system_annotation, user_annotation, self.stopwords)
        turn_annotation['candidates'] = candidates
        turn_annotation['tokens'] = tokens

        return turn_annotation, states

    def _get_state_update(self, current_states, prev_states):
        try:
            prev_states = prev_states["slot_values"]
        finally:
            state_update = dict(current_states)
            for slot, values in current_states.items():
                if slot in prev_states and prev_states[slot][0] in values:
                    state_update.pop(slot)
            return state_update

    def get_candidates(self, system_annotation, user_annotation, stopwords):
        """Candidates include adjs, entities and corefs."""
        tokens = []
        candidates = {}
        entities = []
        postags = []
        corefs = []
        base_index = [0]

        if not self.cfg.extract_sys:
            for sentence in system_annotation["sentences"]:
                tokens.extend([token["word"] for token in sentence["tokens"]])
                base_index.append(base_index[-1] + len(sentence['tokens']))
        else:
            num_sen = 0
            read_annotation(system_annotation, base_index, stopwords, tokens, entities, postags, corefs, num_sen)

        num_sen = len(system_annotation["sentences"])
        read_annotation(user_annotation, base_index, stopwords, tokens, entities, postags, corefs, num_sen)

        candidates['postag'] = postags
        candidates['coref'] = clean(corefs, stopwords)
        candidates['coref'].extend(entities)

        # verify_indices(candidates, tokens)

        return candidates, tokens


def read_annotation(annotation, base_index, stopwords, tokens, entities, postags, corefs, num_sen):
    sentences = annotation["sentences"]
    for i, sentence in enumerate(sentences):

        for entity in sentence['entitymentions']:
            head_idx = base_index[i + num_sen] + entity['tokenBegin']
            head = sentence['tokens'][entity['tokenBegin']]['originalText']
            mention = entity['text']
            mention_start_idx = base_index[i + num_sen] + entity['tokenBegin']
            mention_end_idx = base_index[i + num_sen] + entity['tokenEnd']
            mention_idx = [mention_start_idx, mention_end_idx]
            entities.append([head_idx, head, mention_idx, mention])

        for j, token in enumerate(sentence['tokens']):
            tokens.append(token['word'])
            pos = token['pos']
            lemma = token['lemma']
            text = token['originalText']
            if pos in ['JJ', 'RB']:
                try:
                    prev = sentence['tokens'][j - 1]['originalText']
                except IndexError:
                    prev = ''
                if (not re.search(r"([a-z]\.[a-z])", lemma)
                    ) and lemma not in stopwords and prev != 'not':
                    head_idx = base_index[i + num_sen] + token['index'] - 1
                    flag = True
                    if flag:
                        postags.append(
                            [head_idx, text])

        base_index.append(base_index[-1] + len(sentence['tokens']))

    for coref in annotation['corefs'].values():
        for realization in coref:
            sent_num = realization['sentNum']
            head_index = realization['headIndex']
            head_idx = base_index[sent_num + num_sen] + head_index
            head = sentences[sent_num]['tokens'][head_index]['originalText']
            text_start_index = realization['startIndex']
            text_start_idx = base_index[sent_num + num_sen] + text_start_index
            text_end_index = realization['endIndex']
            text_end_idx = base_index[sent_num + num_sen] + text_end_index
            text_lemma = sentences[sent_num]['tokens'][text_start_index:text_end_index]
            text_lemma = ' '.join(
                list(map(lambda x: x['originalText'], text_lemma)))
            try:
                prev1 = sentences[sent_num]['tokens'][text_start_index -
                                                      1]['originalText']
                prev2 = sentences[sent_num]['tokens'][text_start_index -
                                                      2]['originalText']
            except BaseException:
                prev1 = ''
                prev2 = ''
            if is_stop(
                    text_lemma,
                    stopwords) and prev1 != 'not' and prev2 != 'not':
                flag = True
                if flag:
                    corefs.append([head_idx, head, [text_start_idx, text_end_idx], text_lemma])


def annotate(system_utterance, user_utterance, annotator):
    system_annotation = json.loads(annotator(system_utterance))
    system_annotation['corefs'] = fix_stanford_coref(system_annotation)
    user_annotation = json.loads(annotator(user_utterance))
    user_annotation['corefs'] = fix_stanford_coref(user_annotation)
    return system_annotation, user_annotation


def fix_stanford_coref(stanford_json):
    true_corefs = {}
    # get a chain
    for key, coref in stanford_json["corefs"].items():
        true_coref = []
        # get an entity mention
        for entity in coref:
            sent_num = entity["sentNum"] - 1  # starting from 0
            start_index = entity["startIndex"] - 1  # starting from 0
            end_index = entity["endIndex"] - 1  # starting from 0
            head_index = entity["headIndex"] - 1  # starting from 0
            entity_label = stanford_json["sentences"][
                sent_num]["tokens"][head_index]["ner"]
            entity["sentNum"] = sent_num
            entity["startIndex"] = start_index
            entity["endIndex"] = end_index
            entity["headIndex"] = head_index
            entity["headWord"] = entity["text"].split(
                " ")[head_index - start_index]
            entity["entityType"] = entity_label
            true_coref.append(entity)
        # check link is not empty
        if len(true_coref) > 0:
            no_representative = True
            has_representative = False
            for idx, entity in enumerate(true_coref):
                if entity["isRepresentativeMention"]:
                    if not (entity["type"] == "PRONOMINAL" or
                            bad_entity(entity["text"].lower()) or
                            len(entity["text"].split(" ")) > 10):
                        no_representative = False
                        has_representative = True
                    # remove bad representative assignments
                    else:
                        true_coref[idx]["isRepresentativeMention"] = False
            # check there exists one representative mention
            if no_representative:
                for idx, entity in enumerate(true_coref):
                    if not (entity["type"] == "PRONOMINAL" or
                            bad_entity(entity["text"].lower()) or
                            len(entity["text"].split(" ")) > 10):
                        true_coref[idx]["isRepresentativeMention"] = True
                        has_representative = True
            if has_representative:
                true_corefs[key] = true_coref
    return true_corefs


def verify_indices(all_candidates, tokens):
    for _, candidates in all_candidates.items():
        for candidate in candidates:
            if not candidate[1] == tokens[candidate[0]]:
                print(candidate[1], tokens[candidate[0]],
                      tokens, sep='\n', end='\n\n')


def clean(corefs: list, stopwords: list):
    dup_ids = []
    for i, coref1 in enumerate(corefs):
        consist_num = 0
        short = []
        for j, coref2 in enumerate(corefs):
            if coref1[2][0] <= coref2[2][0] and coref1[2][1] >= coref2[2][1] and (
                    not i == j):
                consist_num += 1
                short.append(j)
        if consist_num > 1:
            dup_ids.append(i)
        elif consist_num == 1:
            dup_ids.extend(short)
    corefs = [corefs[i] for i in range(len(corefs)) if i not in dup_ids]

    temp = []
    for coref in corefs:
        seq = coref[-1].split()
        while seq and (seq[0] in stopwords or seq[-1] in stopwords):
            if seq[0] in stopwords:
                del seq[0]
            if seq[-1] in stopwords:
                del seq[-1]
        if not seq:
            temp.append(coref)
        else:
            coref[-1] = ' '.join(seq)
    for t in temp:
        corefs.remove(t)

    return corefs


def bad_entity(text):
    if text == "this":
        return True
    if text == "that":
        return True
    if text == "there":
        return True
    if text == "here":
        return True
    if text == "|":
        return True
    if text == "less":
        return True
    if text == "more":
        return True
    return False


def is_stop(text: str, stopwords: list):
    text = list(filter(lambda x: x.lower() not in stopwords, text.split()))
    if text:
        return True
    else:
        return False
