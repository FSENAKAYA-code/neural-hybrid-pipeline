# Domain implementation notes

The model supports raw RGB images plus tabular rows through CNN and MLP encoders with a shared cross-entropy objective, as well as a late-fusion head for precomputed embeddings. Splits preserve row alignment; validation selects model state before held-out testing.
