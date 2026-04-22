from __future__ import annotations

import numpy as np

WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4


def wer_list(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    for reference, hypothesis in zip(references, hypotheses):
        result = wer_single(reference, hypothesis)
        total_error += result["num_err"]
        total_del += result["num_del"]
        total_ins += result["num_ins"]
        total_sub += result["num_sub"]
        total_ref_len += result["num_ref"]

    if total_ref_len == 0:
        raise ValueError("references must contain at least one token overall")

    return {
        "wer": (total_error / total_ref_len) * 100,
        "del_rate": (total_del / total_ref_len) * 100,
        "ins_rate": (total_ins / total_ref_len) * 100,
        "sub_rate": (total_sub / total_ref_len) * 100,
    }


def wer_single(reference: str, hypothesis: str):
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    distance = edit_distance(ref_tokens, hyp_tokens)
    alignment, alignment_out = get_alignment(ref_tokens, hyp_tokens, distance)

    num_cor = np.sum([step == "C" for step in alignment])
    num_del = np.sum([step == "D" for step in alignment])
    num_ins = np.sum([step == "I" for step in alignment])
    num_sub = np.sum([step == "S" for step in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(ref_tokens)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": int(num_cor),
        "num_del": int(num_del),
        "num_ins": int(num_ins),
        "num_sub": int(num_sub),
        "num_err": int(num_err),
        "num_ref": int(num_ref),
    }


def edit_distance(reference_tokens, hypothesis_tokens):
    distance = np.zeros(
        (len(reference_tokens) + 1) * (len(hypothesis_tokens) + 1),
        dtype=np.uint16,
    ).reshape((len(reference_tokens) + 1, len(hypothesis_tokens) + 1))

    for i in range(len(reference_tokens) + 1):
        for j in range(len(hypothesis_tokens) + 1):
            if i == 0:
                distance[i][j] = j * WER_COST_INS
            elif j == 0:
                distance[i][j] = i * WER_COST_DEL

    for i in range(1, len(reference_tokens) + 1):
        for j in range(1, len(hypothesis_tokens) + 1):
            if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                substitute = distance[i - 1][j - 1] + WER_COST_SUB
                insert = distance[i][j - 1] + WER_COST_INS
                delete = distance[i - 1][j] + WER_COST_DEL
                distance[i][j] = min(substitute, insert, delete)

    return distance


def get_alignment(reference_tokens, hypothesis_tokens, distance):
    x = len(reference_tokens)
    y = len(hypothesis_tokens)
    max_len = 3 * (x + y)

    align_steps = []
    align_ref = ""
    align_hyp = ""
    align_marks = ""

    while True:
        if (x <= 0 and y <= 0) or (len(align_steps) > max_len):
            break
        if (
            x >= 1
            and y >= 1
            and distance[x][y] == distance[x - 1][y - 1]
            and reference_tokens[x - 1] == hypothesis_tokens[y - 1]
        ):
            align_hyp = " " + hypothesis_tokens[y - 1] + align_hyp
            align_ref = " " + reference_tokens[x - 1] + align_ref
            align_marks = " " * (len(reference_tokens[x - 1]) + 1) + align_marks
            align_steps.append("C")
            x -= 1
            y -= 1
        elif x >= 1 and y >= 1 and distance[x][y] == distance[x - 1][y - 1] + WER_COST_SUB:
            match_len = max(len(hypothesis_tokens[y - 1]), len(reference_tokens[x - 1]))
            align_hyp = " " + hypothesis_tokens[y - 1].ljust(match_len) + align_hyp
            align_ref = " " + reference_tokens[x - 1].ljust(match_len) + align_ref
            align_marks = " S" + " " * (match_len - 1) + align_marks
            align_steps.append("S")
            x -= 1
            y -= 1
        elif y >= 1 and distance[x][y] == distance[x][y - 1] + WER_COST_INS:
            align_hyp = " " + hypothesis_tokens[y - 1] + align_hyp
            align_ref = " " + "*" * len(hypothesis_tokens[y - 1]) + align_ref
            align_marks = " I" + " " * (len(hypothesis_tokens[y - 1]) - 1) + align_marks
            align_steps.append("I")
            y -= 1
        else:
            align_hyp = " " + "*" * len(reference_tokens[x - 1]) + align_hyp
            align_ref = " " + reference_tokens[x - 1] + align_ref
            align_marks = " D" + " " * (len(reference_tokens[x - 1]) - 1) + align_marks
            align_steps.append("D")
            x -= 1

    return align_steps[::-1], {
        "align_ref": align_ref[1:],
        "align_hyp": align_hyp[1:],
        "alignment": align_marks[1:],
    }
