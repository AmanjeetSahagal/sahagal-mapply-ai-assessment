This section focuses mainly on the evaluation results and what they reveal about the system, with a bit of context on how the classification works.

The classifier uses a simple K-NN approach with k set to 5, running over embeddings generated with the MiniLM-L6-v2 model. Each test product is compared to the indexed embeddings, and the five closest neighbors decide the predicted category through majority vote. This setup does not involve any supervised training, so all of the performance comes from the quality of the embeddings and the structure provided by the FAISS indexes.

For evaluation, I computed accuracy, precision, recall, and F1 score for each category, and also generated a full confusion matrix. The FlatL2 index achieved about 44.14% accuracy, while the IVFFlat index came in slightly lower at around 41.22%. This difference is expected because FlatL2 checks all possible neighbors exactly, while IVFFlat uses clustering to speed up searches, which can occasionally return neighbors that are close but not the closest.

Looking deeper into the per class metrics, the performance varies heavily across categories. Categories with clear and meaningful descriptions, such as children’s bikes or school supplies, tend to show high precision and recall because their embeddings cluster tightly together. These categories benefit from having enough representation in the index, which helps the K-NN vote remain consistent.

On the other hand, categories that are rare or overly specific tend to have very low scores. Many of these categories have only a handful of examples or have descriptions that overlap with broader groups. This makes it hard for K-NN to correctly identify their neighbors, which shows up clearly in the confusion matrix where small categories bleed into larger ones.

Overall, the evaluation highlights both the strengths and limitations of this approach. When categories have enough examples and distinct wording, the classifier does well. But in long-tail or ambiguous categories, performance drops, which is expected for a purely similarity‑based method without any supervised learning.
