import json

def compute_accuracy_from_jsonl(file_path):
    total_inform_correct = 0
    total_request_correct = 0
    total_inform_count = 0
    total_request_count = 0

    with open(file_path, "r") as file:
        for line in file:
            example = json.loads(line.strip())
            try:
                if not example["label"].strip():
                    print(example)
                    continue
                label_inform = example["label"].split(",")
                predict_inform = example["predict"].split(",")
                # Compute accuracy for "inform" type
                label_inform_set = set([' '.join(elem.split()[1:]) for elem in label_inform if elem.split()[0] == "inform"])
                predict_inform_set = set([' '.join(elem.split()[1:]) for elem in predict_inform if elem.split()[0] == "inform"])
                inform_correct = len(label_inform_set.intersection(predict_inform_set))
                total_inform_correct += inform_correct
                total_inform_count += len(label_inform_set)

                # Compute accuracy for "request" type
                # label_request_set = set([' '.join(elem.split()[1:]) for elem in label_inform if elem.split()[0] == "request"])
                label_request_set = set([elem.split()[-1] for elem in label_inform if elem.split()[0] == "request"])
                # predict_request_set = set([' '.join(elem.split()[1:]) for elem in predict_inform if elem.split()[0] == "request"])
                predict_request_set = set([elem.split()[-1] for elem in predict_inform if elem.split()[0] == "request"])
                request_correct = len(label_request_set.intersection(predict_request_set))
                total_request_correct += request_correct
                total_request_count += len(label_request_set)
            except:
                print(example)

    inform_accuracy = total_inform_correct / total_inform_count if total_inform_count > 0 else 0
    request_accuracy = total_request_correct / total_request_count if total_request_count > 0 else 0

    return inform_accuracy, request_accuracy


if __name__ == "__main__":
    import sys
    for jsonl_file_path in sys.argv[1:]:
        inform_accuracy, request_accuracy = compute_accuracy_from_jsonl(jsonl_file_path)
        print("Inform Accuracy:", inform_accuracy)
        print("Request Accuracy:", request_accuracy)
    # compute_entity_acc(jsonl_file_path='save/vox-finetune-LLaMAFactory/LLaMA2-7B-Chat/vanilla/generated_predictions.jsonl')
    # compute_entity_acc(jsonl_file_path='save/vox-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_2024-02-06-11-01-44/generated_predictions.jsonl')