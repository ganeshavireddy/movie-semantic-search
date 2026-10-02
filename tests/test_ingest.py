import pandas as pd
from pathlib import Path
import pytest
from src.ingest import clean_and_deduplicate, create_sample_dataset


def test_create_sample_dataset(tmp_path: Path):
    sample_csv = tmp_path / "sample.csv"
    df = create_sample_dataset(sample_csv, num_samples=10)
    assert sample_csv.exists()
    assert len(df) == 10
    assert "Title" in df.columns
    assert "Plot" in df.columns


def test_clean_and_deduplicate():
    raw_data = {
        "Release Year": [2010, 2010, "invalid", 1999],
        "Title": ["Inception", "Inception", "Short Film", "The Matrix"],
        "Origin/Ethnicity": ["American", "American", "British", "American"],
        "Director": ["Christopher Nolan", "Christopher Nolan", "Unknown", "Wachowskis"],
        "Cast": ["Leonardo DiCaprio", "Leonardo DiCaprio", "Actor", "Keanu Reeves"],
        "Genre": ["Sci-Fi", "Sci-Fi", "Drama", "Sci-Fi"],
        "Wiki Page": ["http://wiki/1", "http://wiki/1", "http://wiki/2", "http://wiki/3"],
        "Plot": [
            "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
            "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
            "Too short",  # < 20 chars
            "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
        ]
    }
    df_raw = pd.DataFrame(raw_data)
    df_clean = clean_and_deduplicate(df_raw)

    # Inception deduplicated (1), Short Film dropped (<20 chars), The Matrix kept (1) -> total 2
    assert len(df_clean) == 2
    assert set(df_clean["title"]) == {"Inception", "The Matrix"}
    assert df_clean["release_year"].dtype == "int32" or df_clean["release_year"].dtype == "int64"
