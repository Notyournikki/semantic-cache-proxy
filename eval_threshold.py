import csv
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Each group: first item = the question already in the cache, the rest = paraphrases of it
GROUPS = [
    ["What is machine learning?", "Explain machine learning to me", "Can you tell me what ML is?", "Define machine learning in simple words"],
    ["What is the capital of France?", "Which city is the capital of France?", "Tell me France's capital city", "France's capital is which city?"],
    ["How do I reset my password?", "I forgot my password, how can I change it?", "Steps to recover my account password", "How can I set a new password?"],
    ["How do I delete my account?", "I want to permanently close my account", "How can I remove my profile?", "Steps to deactivate and delete my account"],
    ["What is cloud computing?", "Explain cloud computing simply", "Can you describe what cloud computing means?", "Define cloud computing"],
    ["How does photosynthesis work?", "Explain the process of photosynthesis", "How do plants make food from sunlight?", "Describe photosynthesis step by step"],
    ["What is Docker used for?", "Why do people use Docker?", "Explain what Docker does", "What is the purpose of Docker?"],
    ["How many days are in a leap year?", "How long is a leap year in days?", "Number of days in a leap year?", "A leap year has how many days?"],
    ["Who wrote Romeo and Juliet?", "Who is the author of Romeo and Juliet?", "Romeo and Juliet was written by whom?", "Name the writer of Romeo and Juliet"],
    ["What is the difference between TCP and UDP?", "How do TCP and UDP differ?", "TCP vs UDP explained", "Compare TCP with UDP"],
    ["What is an API?", "Explain what an API is", "Can you describe what an API means?", "Define API in simple terms"],
    ["How do I learn Python quickly?", "What is the fastest way to learn Python?", "Tips for learning Python fast", "How can a beginner pick up Python quickly?"],
]

# Questions that should NOT match anything in the cache (includes tricky look-alikes)
NEGATIVES = [
    "What is deep learning?",
    "What is the capital of Germany?",
    "How do I change my email address?",
    "How do I create a new account?",
    "What is edge computing?",
    "How does respiration work?",
    "What is Kubernetes used for?",
    "How many days are in a normal year?",
    "Who wrote Hamlet?",
    "What is the difference between HTTP and HTTPS?",
    "Best pizza toppings?",
    "What is the weather like today?",
    "How do I bake a chocolate cake?",
    "Who won the last cricket world cup?",
    "Tell me a joke about programmers",
    "What is a database?",
    "How do I learn Java quickly?",
    "What is a REST API?",
    "How do I update my password policy?",
    "What is supervised learning?",
]

cached = [g[0] for g in GROUPS]
cached_vecs = model.encode(cached, normalize_embeddings=True)

queries, truth = [], []   # truth = index of the correct group, or -1 if it should be a MISS
for gi, g in enumerate(GROUPS):
    for p in g[1:]:
        queries.append(p)
        truth.append(gi)
for n in NEGATIVES:
    queries.append(n)
    truth.append(-1)

q_vecs = model.encode(queries, normalize_embeddings=True)
sims = q_vecs @ cached_vecs.T          # cosine similarity matrix
best_idx = sims.argmax(axis=1)
best_score = sims.max(axis=1)

n_para = sum(1 for t in truth if t >= 0)
n_neg = len(truth) - n_para

rows = []
for t in np.arange(0.50, 0.96, 0.05):
    correct = wrong = 0
    for i in range(len(queries)):
        if best_score[i] >= t:
            if truth[i] >= 0 and best_idx[i] == truth[i]:
                correct += 1
            else:
                wrong += 1
    rows.append((round(float(t), 2),
                 round(correct / n_para * 100, 1),
                 round(wrong / len(queries) * 100, 1)))

print(f"{len(queries)} queries: {n_para} paraphrases, {n_neg} should-miss\n")
print("threshold | correct hit rate % | false hit rate %")
for r in rows:
    print(f"   {r[0]:.2f}   |       {r[1]:5.1f}        |      {r[2]:5.1f}")

with open("threshold_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "correct_hit_rate", "false_hit_rate"])
    w.writerows(rows)

plt.figure(figsize=(8, 5))
plt.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label="Correct hit rate %")
plt.plot([r[0] for r in rows], [r[2] for r in rows], marker="s", label="False hit rate %")
plt.xlabel("Similarity threshold")
plt.ylabel("Percent")
plt.title("Threshold vs hit rate and false-hit rate")
plt.legend()
plt.grid(True)
plt.savefig("threshold_graph.png", dpi=150, bbox_inches="tight")
print("\nSaved threshold_graph.png and threshold_results.csv")

order = np.argsort(best_score)
print("\nHardest paraphrases to match (lowest scores):")
shown = 0
for i in order:
    if truth[i] >= 0:
        print(f"  {best_score[i]:.3f}  {queries[i]}")
        shown += 1
        if shown == 5:
            break

print("\nRisky should-miss queries (highest scores = possible false hits):")
shown = 0
for i in order[::-1]:
    if truth[i] < 0:
        print(f"  {best_score[i]:.3f}  {queries[i]}  ->  matched: {cached[best_idx[i]]}")
        shown += 1
        if shown == 5:
            break