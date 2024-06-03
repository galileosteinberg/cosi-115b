'''
Galileo Steinberg
CS115B Spring 2024 Homework 4
Distributional Semantics Takes the SAT
'''

import random

import numpy as np
import scipy
import scipy.linalg as scipy_linalg


random.seed(42)


def compute_co_occurrence(corpus):
    """
    Computes the co-occurrence matrix C, such that
    C[w, c] = the number of (w, c) and (c, w) bigrams in the corpus.
    Multiplies the entire matrix by 10 (to pretend that we see these sentences 10 times) and then smooth the counts by
    adding 1 to all cells.
    """
    words = set(word for sentence in corpus for word in sentence.split())
    word_to_id = {word: i for i, word in enumerate(words)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    co_occurrence_matrix = np.zeros((len(words), len(words)), dtype=int)

    for sentence in corpus:
        tokens = sentence.split()
        for i, token in enumerate(tokens):
            for j in range(max(0, i - 1), min(i + 2, len(tokens))):
                if i != j:
                    co_occurrence_matrix[word_to_id[tokens[i]], word_to_id[tokens[j]]] += 1

    co_occurrence_matrix = co_occurrence_matrix + co_occurrence_matrix.T
    co_occurrence_matrix = 10 * co_occurrence_matrix + 1
    return co_occurrence_matrix, word_to_id, id_to_word


def compute_PPMI(c):
    '''
    Computes the positive pointwise mutual information (PPMI) for each word w and context word c
    '''
    total_sum = c.sum()
    words_sum = c.sum(axis=1, keepdims=True)
    P_w_c = c / total_sum
    P_w = words_sum / total_sum
    P_c = words_sum.T / total_sum
    PPMI = np.maximum(np.log2(P_w_c / (P_w * P_c)), 0)
    PPMI[np.isnan(PPMI)] = 0
    return PPMI


def part1():
    """
    Runs part one of Homework 4.

    Creates the co-occurrence matrix.
    Prints the co-occurrence matrix.

    Creates the ppmi matrix.
    Prints the ppmi matrix.

    Evaluate word similarity with different distance metrics for each word in the word pairs.
    Reduce dimensions with SVD and check the distance metrics on the word pairs again.
    """
    pairs = [
        ("women", "men"),
        ("women", "dogs"),
        ("men", "dogs"),
        ("feed", "like"),
        ("feed", "bite"),
        ("like", "bite"),
    ]
    # TODO: STUDENT CODE HERE
    with open('dist_sim_data.txt', 'r') as file:
        corpus = [line.strip() for line in file if line.strip()]

    C, word_to_id, id_to_word = compute_co_occurrence(corpus)

    PPMI = compute_PPMI(C)
    dogs_index = word_to_id['dogs']
    vector_before = C[dogs_index]
    vector_after = PPMI[dogs_index]

    print("Word vector for 'dogs' before PPMI reweighting:", vector_before)
    print("Word vector for 'dogs' after PPMI reweighting:", vector_after)

    print("\nEuclidean distance between word pairs: ")
    for word1, word2 in pairs:
        idx1, idx2 = word_to_id[word1], word_to_id[word2]
        vector1 = PPMI[idx1, :]
        vector2 = PPMI[idx2, :]
        distance = scipy.linalg.norm(vector1 - vector2)
        print(f"Distance between {word1} and {word2}: {distance}")

    U, E, Vt = scipy_linalg.svd(PPMI, full_matrices=False)
    E = np.diag(E)  # compute E
    print("\n",np.allclose(PPMI, U.dot(E).dot(Vt)))

    V = Vt.T  # compute V = conjugate transpose of Vt
    reduced_PPMI = PPMI.dot(V[:, 0:3])

    print("\nEuclidean distances between word pairs on reduced PPMI matrix:")
    for word1, word2 in pairs:
        idx1, idx2 = word_to_id[word1], word_to_id[word2]
        vector1 = reduced_PPMI[idx1, :]
        vector2 = reduced_PPMI[idx2, :]
        distance = scipy.linalg.norm(vector1 - vector2)
        print(f"Distance between {word1} and {word2} (reduced space): {distance}")


# PART 2.1


def load_word_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors


def load_synonyms_list(file_path):
    synonyms = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # Assuming the first line is a header and should be skipped
        for line in f:
            verb, synonym = line.strip().split('\t')
            # Remove 'to_' prefix from both verb and synonym if present
            verb = verb.replace('to_', '')
            synonym = synonym.replace('to_', '')
            if verb not in synonyms:
                synonyms[verb] = []
            synonyms[verb].append(synonym)
    return synonyms


def create_synonym_test_cases():
    synonyms_list = load_synonyms_list('EN_syn_verb.txt')
    test_cases = []
    verbs = list(synonyms_list.keys())  # All available verbs from synonyms list

    all_possible_non_synonyms = set(verbs)

    mcq_count = 0

    while mcq_count < 1000:
        for verb in verbs:
            if mcq_count >= 1000:
                break
            synonyms = synonyms_list[verb]  # Retrieve synonyms for the current verb

            for synonym in synonyms:
                current_non_synonyms_pool = list(all_possible_non_synonyms - {verb} - set(synonyms))
                if len(current_non_synonyms_pool) < 4: continue

                non_synonyms = random.sample(current_non_synonyms_pool, 4)
                test_cases.append((verb, synonym, non_synonyms))
                mcq_count += 1
                if mcq_count == 1000: break

    with open('synonym_test_set.txt', 'w', encoding='utf-8') as f:
        for verb, correct_synonym, distractors in test_cases:
            distractors_str = ' '.join(distractors)
            line = f"{verb} {correct_synonym} {distractors_str}\n"
            f.write(line)


def read_synonym_test_cases_from_file(file_path):
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            verb = parts[0]
            correct_synonym = parts[1]
            distractors = parts[2:]
            test_cases.append((verb, correct_synonym, distractors))
    return test_cases


def compute_euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1-vec2)


def compute_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def evaluate_test_cases(test_cases, word_vectors):
    correct_euclidean, correct_cosine = 0, 0
    num_evaluated = 0

    for verb, correct_synonym, distractors in test_cases:
        # Perform the evaluation only if the verb and correct synonym are in the word vectors
        if verb in word_vectors and correct_synonym in word_vectors:
            num_evaluated += 1
            verb_vector = word_vectors[verb]
            correct_vector = word_vectors[correct_synonym]

            choices = distractors + [correct_synonym]
            choices_vectors = [(choice, word_vectors.get(choice, None)) for choice in choices if choice in word_vectors]

            # Compute Euclidean distances and cosine similarities
            distances = [(choice[0], compute_euclidean_distance(verb_vector, choice[1])) for choice in choices_vectors]
            similarities = [(choice[0], compute_cosine_similarity(verb_vector, choice[1])) for choice in choices_vectors]

            # Identify the most similar choice to the verb's vector
            closest_by_distance = min(distances, key=lambda x: x[1])[0]
            closest_by_similarity = max(similarities, key=lambda x: x[1])[0]

            # Increment correct counters
            if closest_by_distance == correct_synonym:
                correct_euclidean += 1
            if closest_by_similarity == correct_synonym:
                correct_cosine += 1

    # Print results
    accuracy_euclidean = correct_euclidean / num_evaluated if num_evaluated else 0
    accuracy_cosine = correct_cosine / num_evaluated if num_evaluated else 0
    print(f"Accuracy using Euclidean distance: {accuracy_euclidean:.2%}")
    print(f"Accuracy using Cosine similarity: {accuracy_cosine:.2%}")


def run_synonym_test():
    create_synonym_test_cases()

    # Load both Google's word2vec and COMPOSES word vectors
    google_word_vectors = load_word_vectors('GoogleNews-vectors-negative300-filtered.txt')
    composes_word_vectors = load_word_vectors('EN-wform.w.2.ppmi.svd.500-filtered.txt')

    test_cases = read_synonym_test_cases_from_file('synonym_test_set.txt')

    print("\nSynonym Test Evaluation with word2vec:")
    evaluate_test_cases(test_cases, google_word_vectors)

    print("\nSynonym Test Evaluation with COMPOSES:")
    evaluate_test_cases(test_cases, composes_word_vectors)


# PART 2.2


def load_sat_questions(file_path):
    """Load SAT analogy questions, considering block separation, source lines, and word separation."""
    filtered_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # Split content into blocks separated by blank lines
        for line in file:
            if not line.startswith('#'):
                filtered_lines.append(line.rstrip())

    content = '\n'.join(filtered_lines)  # Reconstruct content without comment lines
    content_blocks = content.strip().split('\n\n')

    sat_questions = []
    for block in content_blocks:
        lines = [line.rstrip() for line in block.split('\n') if not line.startswith('#')][1:]
        stem = lines[0][:-4].split()  # Separate words in the stem line, discard POS
        choices = [line[:-4].split() for line in lines[1:6]]  # Separate words in choice lines, discard POS
        correct_choice = lines[-1].strip()  # The correct choice is indicated by the last line

        sat_questions.append({
            'stem': stem,  # This will be a list of two words [word1, word2]
            'choices': choices,  # A list of pairs, each being a list of two words [[word1, word2], ...]
            'correct_letter': correct_choice
        })

    return sat_questions


def solve_analogy(stem, choices, word_vectors):
    # Attempt to retrieve vectors for the stem words. Skip this question if any vector is missing.
    if stem[0] not in word_vectors or stem[1] not in word_vectors:
        return None
    stem_vec_diff = word_vectors[stem[0]] - word_vectors[stem[1]]

    best_choice = None
    max_similarity = float('-inf')

    # Loop through the choices to find the one most similar to the stem's relation vector
    for i, (choice_a, choice_b) in enumerate(choices):
        if choice_a not in word_vectors or choice_b not in word_vectors:
            continue  # Skip choices with missing vectors
        choice_vec_diff = word_vectors[choice_a] - word_vectors[choice_b]
        similarity = compute_cosine_similarity(stem_vec_diff, choice_vec_diff)
        if similarity > max_similarity:
            max_similarity = similarity
            best_choice = chr(ord('a') + i)

    return best_choice


def evaluate_sat(sat_questions, word_vectors, vectors_name):
    correct, total = 0, 0
    for question in sat_questions:
        answer = solve_analogy(question['stem'], question['choices'], word_vectors)
        if answer and answer == question['correct_letter']:
            correct += 1
        if answer:  # Only consider questions where an answer could be generated
            total += 1

    # Calculate and print accuracy directly within the function
    accuracy = correct / total if total > 0 else 0
    print(f"{vectors_name} Accuracy: {accuracy:.2%}")


def run_sat_test():
    """
    Sets up the SAT test, loads the word embeddings and runs the evaluation.
    Prints the overall accuracy of the SAT task.
    """
    # TODO: STUDENT CODE HERE
    google_word_vectors = load_word_vectors('GoogleNews-vectors-negative300-filtered.txt')
    composes_word_vectors = load_word_vectors('EN-wform.w.2.ppmi.svd.500-filtered.txt')
    sat_questions = load_sat_questions('SAT-package-V3.txt')

    # Evaluate SAT questions for word2vec
    print("\nSAT Analogy Test Evaluation with word2vec:")
    evaluate_sat(sat_questions, google_word_vectors, "word2vec")

    # Evaluate SAT questions for COMPOSES
    print("\nSAT Analogy Test Evaluation with COMPOSES:")
    evaluate_sat(sat_questions, composes_word_vectors, "COMPOSES")


def part2():
    """
    Runs the two tasks for part two of Homework 4.
    """

    run_synonym_test()
    run_sat_test()


if __name__ == "__main__":
    # DO NOT MODIFY HERE
    part1()
    part2()