# DNA Sequence Encoding and Classification

This project explores different encoding methods to classify DNA sequences, aiming to find the most effective approach for predictive modeling. By transforming DNA sequences through di-nucleotide, tri-nucleotide, and Fourier transform encodings, this project builds and evaluates classification models using a Random Forest classifier.

## Objective
To identify the most effective encoding method for classifying DNA sequences based on di-nucleotide, tri-nucleotide, and Fourier transform encodings.

## Dataset
The dataset contains DNA sequences and their associated class labels, which represent different biological categories. It is loaded from a TSV file.

## Project Structure
- `dna_sequence_classification.py`: Main Python file containing data processing, feature engineering, model training, and evaluation code.
- `README.md`: Project overview and instructions (this file).
- `data`: Folder for storing the input dataset file (`extraA3_Data.tsv`).

## Feature Engineering and Encodings
1. **Di-nucleotide Encoding**: Calculates the frequencies of each di-nucleotide (two-nucleotide sequence) within each DNA sequence.
2. **Tri-nucleotide Encoding**: Calculates the frequencies of each tri-nucleotide (three-nucleotide sequence) within each DNA sequence.
3. **Fourier Transform Encoding**: Maps each DNA sequence to numerical values and applies the Fourier Transform, capturing frequency information.

## Model and Evaluation
A Random Forest classifier is trained and evaluated using 10-fold stratified cross-validation on each encoding method. Key evaluation metrics include F1-score and Mean Average Precision (AP).

### Results Summary
| Encoding Method          | Mean F1 Score | Std. Dev. F1 Score | Mean Average Precision | Std. Dev. Average Precision |
|--------------------------|---------------|---------------------|-------------------------|-----------------------------|
| **Di-nucleotide**        | 0.6148        | 0.0418             | 0.5475                 | 0.0615                      |
| **Tri-nucleotide**       | 0.5324        | 0.0751             | 0.5372                 | 0.0971                      |
| **Fourier Transform**    | 0.4278        | 0.0015             | 0.2834                 | 0.0458                      |

## Precision-Recall Analysis
The following precision-recall curves display the performance of each encoding method across all folds, highlighting the effectiveness of di-nucleotide and tri-nucleotide encodings.

- **Di-Nucleotide Encoding Precision-Recall Curve**  
  ![Di-Nucleotide Precision-Recall Curve](Di-Nucleotide%20Encoding%20Precision-Recall%20Curve.png)

- **Tri-Nucleotide Encoding Precision-Recall Curve**  
  ![Tri-Nucleotide Precision-Recall Curve](Tri-Nucleotide%20Encoding%20Precision-Recall%20Curve.png)

- **Fourier Transform Encoding Precision-Recall Curve**  
  ![Fourier Transform Precision-Recall Curve](Fourier%20Transform%20Encoding%20Precision-Recall%20Curve.png)

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/dna-sequence-classification.git
    cd dna-sequence-classification
    ```

2. **Install required packages**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib scipy
    ```

3. **Run the code**:
    ```bash
    python ea3.py
    ```

4. **Results**:
   - Outputs F1-scores and Average Precision for each encoding method.
   - Generates precision-recall curves for visual comparison.

## Insights
1. **Di-nucleotide Encoding**: Achieved the highest mean F1 score and precision, demonstrating strong performance in sequence classification.
2. **Tri-nucleotide Encoding**: Performed comparably well, showing potential for further exploration.
3. **Fourier Transform Encoding**: Lower scores indicate it may not capture DNA sequence patterns as effectively for this task.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
