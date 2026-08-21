"""A small, hand-built retrieval evaluation dataset.

14 short passages spanning: SMOTE/imbalanced-learning (this project's
running example), a few similar-but-distinct ML techniques (to test
discrimination against near-topic confusion), and several unrelated
distractor topics (to test that retrieval doesn't get confused by
surface-level noise).

22 questions, each with the id(s) of the passage(s) that answer it.
Deliberately includes:
- questions built around rare, exact tokens (BM25's strength),
- questions that paraphrase the source passage with almost no shared
  vocabulary (semantic/vector search's strength),
- questions distinguishing between topically-similar passages
  (random forest vs. gradient boosting), to test whether retrieval
  actually discriminates rather than just finding "something ML-ish".
"""
from dataclasses import dataclass

CORPUS: dict[str, str] = {
    "smote_def": (
        "SMOTE (Synthetic Minority Over-sampling Technique) addresses class "
        "imbalance by creating new synthetic examples of the rare class rather "
        "than simply duplicating existing ones. For each minority-class point, "
        "it finds its nearest neighbors within the same class and generates "
        "new points along the line segments connecting them."
    ),
    "smote_algorithm_detail": (
        "To generate a synthetic sample, SMOTE picks a minority-class point, "
        "randomly selects one of its k nearest minority-class neighbors, and "
        "creates a new point at a random position along the line connecting "
        "the two, effectively interpolating between real examples."
    ),
    "smote_disadvantage": (
        "A known drawback of SMOTE is that it can introduce noisy or "
        "borderline synthetic samples when minority-class instances are "
        "scattered near the majority-class boundary, sometimes making the "
        "decision boundary less clean and increasing the risk of overfitting."
    ),
    "smote_vs_undersampling": (
        "Random under-sampling removes majority-class examples until the "
        "classes are balanced, which is simple but discards potentially "
        "useful data. Combining SMOTE's synthetic over-sampling with a "
        "moderate amount of majority-class under-sampling often yields better "
        "classifier performance than either technique alone."
    ),
    "smote_evaluation": (
        "Because plain accuracy is misleading on imbalanced datasets, "
        "practitioners typically evaluate resampling techniques like SMOTE "
        "using the ROC curve and the area under it (AUC), which separately "
        "track the true positive rate and false positive rate across "
        "different decision thresholds."
    ),
    "random_forest": (
        "A random forest is an ensemble of decision trees, each trained on a "
        "bootstrap sample of the data with a random subset of features "
        "considered at each split. Predictions are combined by majority vote "
        "or averaging, which reduces the variance of any single tree."
    ),
    "gradient_boosting": (
        "Gradient boosting builds an ensemble of shallow decision trees "
        "sequentially, where each new tree is trained to predict the residual "
        "errors of the combined ensemble so far, gradually reducing the "
        "overall loss."
    ),
    "cross_validation": (
        "K-fold cross-validation splits the training data into k roughly "
        "equal parts, trains the model k times leaving out one fold each time "
        "for validation, and averages the resulting performance estimates to "
        "reduce the variance of a single train/test split."
    ),
    "neural_net_training": (
        "Training a neural network with stochastic gradient descent updates "
        "the model's weights using small batches of data at a time, using "
        "the gradient of the loss function to nudge each weight in the "
        "direction that reduces error."
    ),
    "qz7_widget": (
        "The QZ7-Widget's calibration routine requires exactly 42 preheating "
        "cycles before the firmware will accept a new sensor profile."
    ),
    "sourdough_bread": (
        "Sourdough bread relies on a naturally fermented starter culture of "
        "wild yeast and lactic acid bacteria, which is fed regularly with "
        "flour and water to keep it active before it's mixed into the dough."
    ),
    "roman_aqueducts": (
        "Roman aqueducts used gravity alone to move water over long "
        "distances, relying on a very gradual, carefully engineered downward "
        "slope rather than pumps, often crossing valleys on tall stone arches."
    ),
    "guitar_tuning": (
        "Standard guitar tuning, from the lowest to the highest string, is "
        "E-A-D-G-B-E, and a chromatic tuner listens to the pitch of a plucked "
        "string to show how far it is from the nearest correct note."
    ),
    "pour_over_coffee": (
        "Pour-over coffee brewing controls extraction by pouring hot water "
        "slowly and evenly over medium-ground coffee in a filter, allowing "
        "gravity to draw the liquid through at a controlled rate."
    ),
}


@dataclass
class EvalQuestion:
    question: str
    relevant_ids: list[str]
    note: str = ""  # what this question is designed to probe


QUESTIONS: list[EvalQuestion] = [
    # --- Rare exact-token questions: BM25's strength ---
    EvalQuestion("What does the QZ7-Widget calibration routine require?", ["qz7_widget"], "rare-token"),
    EvalQuestion("How many preheating cycles does the QZ7-Widget need?", ["qz7_widget"], "rare-token"),
    EvalQuestion("What is SMOTE?", ["smote_def"], "exact-term"),
    EvalQuestion("What does SMOTE stand for?", ["smote_def"], "exact-term"),
    # --- Semantic paraphrases sharing almost no vocabulary: vector search's strength ---
    EvalQuestion(
        "How can you fix a machine learning dataset where one category has far fewer examples than another?",
        ["smote_def"],
        "paraphrase",
    ),
    EvalQuestion(
        "What downsides come from generating extra fake data points for a rare outcome category?",
        ["smote_disadvantage"],
        "paraphrase",
    ),
    EvalQuestion(
        "Which metric should I use instead of plain accuracy when one class is much rarer than the other?",
        ["smote_evaluation"],
        "paraphrase",
    ),
    EvalQuestion(
        "How do you combine deleting some common-category rows with inventing new rare-category rows?",
        ["smote_vs_undersampling"],
        "paraphrase",
    ),
    EvalQuestion(
        "Why is accuracy a bad metric when most examples belong to one class?",
        ["smote_evaluation"],
        "paraphrase",
    ),
    # --- General topic questions ---
    EvalQuestion("What is a random forest?", ["random_forest"], "general"),
    EvalQuestion("How does gradient boosting work?", ["gradient_boosting"], "general"),
    EvalQuestion("What is k-fold cross-validation?", ["cross_validation"], "general"),
    EvalQuestion("How are neural network weights updated during training?", ["neural_net_training"], "general"),
    EvalQuestion("How does SMOTE decide where to place a new synthetic point?", ["smote_algorithm_detail"], "general"),
    EvalQuestion(
        "What's the difference between SMOTE and just deleting majority-class rows?",
        ["smote_vs_undersampling"],
        "general",
    ),
    # --- Discrimination between topically-similar passages ---
    EvalQuestion(
        "What ensemble method trains trees one after another to fix previous mistakes?",
        ["gradient_boosting"],
        "discrimination",
    ),
    EvalQuestion(
        "What ensemble method trains many trees independently and averages their votes?",
        ["random_forest"],
        "discrimination",
    ),
    EvalQuestion(
        "How many times is a model retrained when using 5-fold cross-validation?",
        ["cross_validation"],
        "discrimination",
    ),
    # --- Distractor topics (unrelated to ML) ---
    EvalQuestion("What keeps a sourdough starter alive?", ["sourdough_bread"], "distractor"),
    EvalQuestion("How did Roman aqueducts move water without pumps?", ["roman_aqueducts"], "distractor"),
    EvalQuestion("What is standard guitar tuning?", ["guitar_tuning"], "distractor"),
    EvalQuestion("How does pour-over coffee brewing control extraction?", ["pour_over_coffee"], "distractor"),
]
