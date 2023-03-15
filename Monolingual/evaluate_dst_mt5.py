#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Function : 
@License : Copyright(C), ILPS group, Univeristy of Amsterdam
@Author  : Jiahuan Pei
@Contact : j.pei@uva.nl
@Date: 3/26/21
"""
import json
from train_mt5 import setup_train_args
def compare_request_lists(list_a, list_b):

    if len(list_a) != len(list_b):
        return False

    list_a.sort()
    list_b.sort()

    for idx in range(0, len(list_a)):
        if list_a[idx] != list_b[idx]:
            return False

    return True

def evaluate_woz(evaluated_dialogues, dialogue_ontology):
    """
    Given a list of (transcription, correct labels, predicted labels), this measures joint goal (as in Matt's paper),
    and f-scores, as presented in Shawn's NIPS paper.

    Assumes request is always there in the ontology.
    """
    global true_value, predicted_value, goal_joint_total
    print_mode = True
    informable_slots = list({"food", "area", "price range", "prezzo", "cibo", "essen", "preisklasse", "gegend"} & set(
            dialogue_ontology.keys()))
    dialogue_count = len(evaluated_dialogues)
    if "request" in dialogue_ontology:
        req_slots = [str("req_" + x) for x in dialogue_ontology["request"]]
        requestables = ["request"]
    else:
        req_slots = []
        requestables = []
    # print req_slots

    true_positives = {}
    false_negatives = {}
    false_positives = {}

    req_match = 0.0
    req_full_turn_count = 0.0

    req_acc_total = 0.0  # number of turns which express requestables
    req_acc_correct = 0.0

    for slot in dialogue_ontology:
        true_positives[slot] = 0
        false_positives[slot] = 0
        false_negatives[slot] = 0

    #false_positives['req_dontcare'] = 0
    for value in requestables + req_slots + ["request"]:
        true_positives[value] = 0
        false_positives[value] = 0
        false_negatives[value] = 0

    correct_turns = 0  # when there is at least one informable, do all of them match?
    incorrect_turns = 0  # when there is at least one informable, if any are different.

    slot_correct_turns = {}
    slot_incorrect_turns = {}

    for slot in informable_slots:
        slot_correct_turns[slot] = 0.0
        slot_incorrect_turns[slot] = 0.0

    dialogue_joint_metrics = []
    dialogue_req_metrics = []

    dialogue_slot_metrics = {}

    for slot in informable_slots:
        dialogue_slot_metrics[slot] = []

    for idx in range(0, dialogue_count):

        dialogue = evaluated_dialogues[idx]["dialogue"]
        # print dialogue

        curr_dialogue_goal_joint_total = 0.0  # how many turns have informables
        curr_dialogue_goal_joint_correct = 0.0

        curr_dialogue_goal_slot_total = {}  # how many turns in current dialogue have specific informables
        curr_dialogue_goal_slot_correct = {}  # and how many of these are correct

        for slot in informable_slots:
            curr_dialogue_goal_slot_total[slot] = 0.0
            curr_dialogue_goal_slot_correct[slot] = 0.0

        creq_tp = 0.0
        creq_fn = 0.0
        creq_fp = 0.0
        # to compute per-dialogue f-score for requestables

        for turn in dialogue:

            # first update full requestable

            req_full_turn_count += 1.0

            if requestables:

                if compare_request_lists(turn[1]["True State"]["request"], turn[2]["Prediction"]["request"]):
                    req_match += 1.0

                if len(turn[1]["True State"]["request"]) > 0:
                    req_acc_total += 1.0

                    if compare_request_lists(turn[1]["True State"]["request"], turn[2]["Prediction"]["request"]):
                        req_acc_correct += 1.0

            # per dialogue requestable metrics
            if requestables:

                true_requestables = turn[1]["True State"]["request"]
                predicted_requestables = turn[2]["Prediction"]["request"]

                for each_true_req in true_requestables:
                    if each_true_req in dialogue_ontology["request"] and each_true_req in predicted_requestables:
                        true_positives["request"] += 1
                        creq_tp += 1.0
                        true_positives["req_" + each_true_req] += 1
                    elif each_true_req in dialogue_ontology["request"]:
                        false_negatives["request"] += 1
                        false_negatives["req_" + each_true_req] += 1
                        creq_fn += 1.0
                        # print "FN:", turn[0], "---", true_requestables, "----", predicted_requestables

                for each_predicted_req in predicted_requestables:
                    # ignore matches, already counted, now need just negatives:
                    if each_predicted_req not in true_requestables :
                        false_positives["request"] += 1
                        # guojun add
                        if "req_" + each_predicted_req in false_positives.keys():
                            false_positives["req_" + each_predicted_req] += 1
                        else:
                            false_positives["req_" + each_predicted_req] = 0
                        creq_fp += 1.0
                        # print "-- FP:", turn[0], "---", true_requestables, "----", predicted_requestables

            # print turn
            inf_present = {}
            inf_correct = {}

            for slot in informable_slots:
                inf_present[slot] = False
                inf_correct[slot] = True

            informable_present = False
            informable_correct = True

            for slot in informable_slots:

                try:
                    true_value = turn[1]["True State"][slot]
                    predicted_value = turn[2]["Prediction"][slot]
                except:

                    print("PROBLEM WITH", turn, "slot:", slot, "inf slots", informable_slots)

                if true_value != "none":
                    informable_present = True
                    inf_present[slot] = True

                if true_value == predicted_value:  # either match or none, so not incorrect
                    if true_value != "none":
                        true_positives[slot] += 1
                else:
                    if true_value == "none":
                        false_positives[slot] += 1
                    elif predicted_value == "none":
                        false_negatives[slot] += 1
                    else:
                        # spoke to Shawn - he does this as false negatives for now - need to think about how we evaluate it properly.
                        false_negatives[slot] += 1

                    informable_correct = False
                    inf_correct[slot] = False

            if informable_present:

                curr_dialogue_goal_joint_total += 1.0

                if informable_correct:
                    correct_turns += 1
                    curr_dialogue_goal_joint_correct += 1.0
                else:
                    incorrect_turns += 1

            for slot in informable_slots:
                if inf_present[slot]:
                    curr_dialogue_goal_slot_total[slot] += 1.0

                    if inf_correct[slot]:
                        slot_correct_turns[slot] += 1.0
                        curr_dialogue_goal_slot_correct[slot] += 1.0
                    else:
                        slot_incorrect_turns[slot] += 1.0

        # current dialogue requestables

        if creq_tp + creq_fp > 0.0:
            creq_precision = creq_tp / (creq_tp + creq_fp)
        else:
            creq_precision = 0.0

        if creq_tp + creq_fn > 0.0:
            creq_recall = creq_tp / (creq_tp + creq_fn)
        else:
            creq_recall = 0.0

        if creq_precision + creq_recall == 0:
            if creq_tp == 0 and creq_fn == 0 and creq_fn == 0:
                # no requestables expressed, special value
                creq_fscore = -1.0
            else:
                creq_fscore = 0.0  # none correct but some exist
        else:
            creq_fscore = (2 * creq_precision * creq_recall) / (creq_precision + creq_recall)

        dialogue_req_metrics.append(creq_fscore)

        # and current dialogue informables:

        for slot in informable_slots:
            if curr_dialogue_goal_slot_total[slot] > 0:
                dialogue_slot_metrics[slot].append(
                    float(curr_dialogue_goal_slot_correct[slot]) / curr_dialogue_goal_slot_total[slot])
            else:
                dialogue_slot_metrics[slot].append(-1.0)

        if informable_slots:
            if curr_dialogue_goal_joint_total > 0:
                current_dialogue_joint_metric = float(curr_dialogue_goal_joint_correct) / curr_dialogue_goal_joint_total
                dialogue_joint_metrics.append(current_dialogue_joint_metric)
            else:
                # should not ever happen when all slots are used, but for validation we might not have i.e. area mentioned
                dialogue_joint_metrics.append(-1.0)

    if informable_slots:
        goal_joint_total = float(correct_turns) / float(correct_turns + incorrect_turns)

    slot_gj = {}

    total_true_positives = 0
    total_false_negatives = 0
    total_false_positives = 0

    precision = {}
    recall = {}
    fscore = {}

    # FSCORE for each requestable slot:
    if requestables:
        add_req = ["request"] + req_slots
    else:
        add_req = []

    for slot in informable_slots + add_req:

        if slot not in ["request"] and slot not in req_slots:
            total_true_positives += true_positives[slot]
            total_false_positives += false_positives[slot]
            total_false_negatives += false_negatives[slot]

        precision_denominator = (true_positives[slot] + false_positives[slot])

        if precision_denominator != 0:
            precision[slot] = float(true_positives[slot]) / precision_denominator
        else:
            precision[slot] = 0

        recall_denominator = (true_positives[slot] + false_negatives[slot])

        if recall_denominator != 0:
            recall[slot] = float(true_positives[slot]) / recall_denominator
        else:
            recall[slot] = 0

        if precision[slot] + recall[slot] != 0:
            fscore[slot] = (2 * precision[slot] * recall[slot]) / (precision[slot] + recall[slot])
            print("REQ - slot", slot, round(precision[slot], 3), round(recall[slot], 3), round(fscore[slot], 3))
        else:
            fscore[slot] = 0

        total_count_curr = true_positives[slot] + false_negatives[slot] + false_positives[slot]

        # if "req" in slot:
        # if slot in ["area", "food", "price range", "request"]:
        # print "Slot:", slot, "Count:", total_count_curr, true_positives[slot], false_positives[slot], false_negatives[slot], "[Precision, Recall, Fscore]=", round(precision[slot], 2), round(recall[slot], 2), round(fscore[slot], 2)
        # print "Slot:", slot, "TP:", true_positives[slot], "FN:", false_negatives[slot], "FP:", false_positives[slot]

    if requestables:

        requested_accuracy_all = req_match / req_full_turn_count

        if req_acc_total != 0:
            requested_accuracy_exist = req_acc_correct / req_acc_total
        else:
            requested_accuracy_exist = 1.0

        slot_gj["request"] = round(requested_accuracy_exist, 3)
        # slot_gj["requestf"] = round(fscore["request"], 3)

    for slot in informable_slots:
        slot_gj[slot] = round(
            float(slot_correct_turns[slot]) / float(slot_correct_turns[slot] + slot_incorrect_turns[slot]), 3)

    # NIKOLA
    if len(informable_slots) == 3:
        # print "\n\nGoal Joint: " + str(round(goal_joint_total, 3)) + "\n"
        slot_gj["joint"] = round(goal_joint_total, 3)

    #if "request" in slot_gj:
    #   del slot_gj["request"]

    return slot_gj

if __name__ == "__main__":
    args = setup_train_args()
    evaluated_dialogues = json.load(open(args.save_path, 'r', encoding='utf-8'))
    ontology = json.load(open(args.ontology_path, 'r', encoding='utf-8'))["informable"]
    results = evaluate_woz(evaluated_dialogues, ontology)
    print(results)
    pass