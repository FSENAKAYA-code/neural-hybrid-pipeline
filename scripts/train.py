"""Run a small deterministic demo; no external data is fetched."""

from sklearn.datasets import make_classification

from hybrid.experiment import run_experiment


def main() -> None:
    X, y = make_classification(
        n_samples=500, n_features=12, n_informative=8, random_state=42
    )
    print(run_experiment(X, y, random_state=42))


if __name__ == "__main__":
    main()
