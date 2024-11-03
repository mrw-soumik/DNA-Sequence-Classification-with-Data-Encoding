import pandas as pd
import numpy as np
from scipy.fft import fft
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Load the data
file_path = 'extraA3_Data.tsv'  # Adjust path if needed
data = pd.read_csv(file_path, sep='\t', header=None, names=['sequence', 'label'])

# Helper functions
def get_kmer_frequencies(sequence, k=2):
    k_mers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    total_kmers = len(k_mers)
    kmer_counts = Counter(k_mers)
    return {kmer: count / total_kmers for kmer, count in kmer_counts.items()}

def numerical_mapping(sequence):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
    return [mapping.get(nucleotide, 0) for nucleotide in sequence]

def apply_fourier_transform(numerical_sequence):
    transformed = fft(numerical_sequence)
    return np.abs(transformed)

# Apply encodings
data['di_nucleotide_freqs'] = data['sequence'].apply(lambda seq: get_kmer_frequencies(seq, k=2))
data['tri_nucleotide_freqs'] = data['sequence'].apply(lambda seq: get_kmer_frequencies(seq, k=3))
data['numerical_sequence'] = data['sequence'].apply(numerical_mapping)
data['fourier_transform'] = data['numerical_sequence'].apply(apply_fourier_transform)

# Extract features
di_nucleotides = [a + b for a in 'ACGT' for b in 'ACGT']
tri_nucleotides = [a + b + c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']
max_length_fourier = max(len(ft) for ft in data['fourier_transform'])

X_di_nucleotide = np.array(data['di_nucleotide_freqs'].apply(lambda freqs: [freqs.get(kmer, 0) for kmer in di_nucleotides]).tolist())
X_tri_nucleotide = np.array(data['tri_nucleotide_freqs'].apply(lambda freqs: [freqs.get(kmer, 0) for kmer in tri_nucleotides]).tolist())
X_fourier = np.array([np.pad(ft, (0, max_length_fourier - len(ft)), 'constant') for ft in data['fourier_transform']])
y = data['label'].values

# Model Evaluation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
clf = RandomForestClassifier(random_state=42)

def evaluate_model(X, y, classifier, cv):
    f1_scores = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    return np.mean(f1_scores), np.std(f1_scores)

mean_f1_di, std_f1_di = evaluate_model(X_di_nucleotide, y, clf, cv)
mean_f1_tri, std_f1_tri = evaluate_model(X_tri_nucleotide, y, clf, cv)
mean_f1_fourier, std_f1_fourier = evaluate_model(X_fourier, y, clf, cv)

print(f"Di-nucleotide F1: Mean = {mean_f1_di:.4f}, Std. = {std_f1_di:.4f}")
print(f"Tri-nucleotide F1: Mean = {mean_f1_tri:.4f}, Std. = {std_f1_tri:.4f}")
print(f"Fourier Transform F1: Mean = {mean_f1_fourier:.4f}, Std. = {std_f1_fourier:.4f}")

# Plotting Precision-Recall Curves
def evaluate_precision_recall(X, y, classifier, cv):
    precision_list, recall_list, avg_precision_list = [], [], []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_scores = classifier.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        avg_precision = average_precision_score(y_test, y_scores)
        precision_list.append(precision)
        recall_list.append(recall)
        avg_precision_list.append(avg_precision)
    return precision_list, recall_list, avg_precision_list

def plot_precision_recall(precision, recall, avg_precision, title):
    plt.figure(figsize=(7, 5))
    for i in range(len(precision)):
        plt.plot(recall[i], precision[i], lw=2, alpha=0.3, label=f'Fold {i+1} (AP={avg_precision[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.show()

precision_di, recall_di, ap_di = evaluate_precision_recall(X_di_nucleotide, y, clf, cv)
plot_precision_recall(precision_di, recall_di, ap_di, 'Di-Nucleotide Encoding Precision-Recall')
