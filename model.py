import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# DATA PREPARATION 
rna_train = pd.read_table('C:\\Users\\swani2\\Pictures\\paper_projects\\lnc\\data\\mRNA_lncRNA_random_data_set.txt')

def get_kmers(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

rna_train['words'] = rna_train.apply(lambda x: " ".join(get_kmers(x['sequence'])), axis=1)
X = CountVectorizer().fit_transform(rna_train['words'])
y = rna_train['class'].values


models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=7),
    "ANN (predictLNCrna)": MLPClassifier(solver='lbfgs', alpha=1e-5, 
                                        hidden_layer_sizes=(6,), random_state=1)
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)
results_list = []
plot_data = {} # To store probabilities for ROC/PR curves

for name, clf in models.items():
    print(f"Processing {name}...")
    
    
    y_probs = cross_val_predict(clf, X, y, cv=kf, method='predict_proba')[:, 1]
    y_pred = (y_probs > 0.5).astype(int)
    
    
    plot_data[name] = y_probs
    
    
    results_list.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1-Score": f1_score(y, y_pred),
        "AUC-ROC": auc(*roc_curve(y, y_probs)[:2])
    })

df_results = pd.DataFrame(results_list)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)


ax1 = fig.add_subplot(gs[0, 0])
for name, probs in plot_data.items():
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right')


ax2 = fig.add_subplot(gs[0, 1])
for name, probs in plot_data.items():
    precision, recall, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    ax2.plot(recall, precision, label=f'{name} (AP = {ap:.2f})')
ax2.set_title('Precision-Recall Curves', fontsize=14)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend(loc='lower left')


ax3 = fig.add_subplot(gs[1, :])
df_melted = df_results.melt(id_vars="Algorithm", var_name="Metric", value_name="Score")
sns.barplot(data=df_melted, x="Algorithm", y="Score", hue="Metric", palette="magma", ax=ax3)
ax3.set_title('Global Performance Comparison', fontsize=14)
ax3.set_ylim(0, 1.1)
ax3.legend(bbox_to_anchor=(1, 1))

plt.show()


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, (name, probs) in enumerate(plot_data.items()):
    cm = confusion_matrix(y, (probs > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[i], cbar=False)
    axes[i].set_title(f'Confusion Matrix: {name}')
plt.tight_layout()
plt.show()

print(df_results)
