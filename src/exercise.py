import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def main():
    # LGetting the absolute pth of exercise.py file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Getting the path to the 'data' folder, which contains this file
    data_path = os.path.normpath(os.path.join(script_dir, "..", "data"))

    # read_csv is used to read all of the 5 data files
    df_externalfactors = pd.read_csv(os.path.join(data_path, 'external_factors.csv'))
    df_matches = pd.read_csv(os.path.join(data_path, 'matches.csv'))
    df_players = pd.read_csv(os.path.join(data_path, 'players.csv'))
    df_teams = pd.read_csv(os.path.join(data_path, 'teams.csv'))
    df_venues = pd.read_csv(os.path.join(data_path, 'venues.csv'))

    # Storing the dataframes in a list
    data_files = [df_externalfactors, df_matches, df_players, df_teams, df_venues]

    for data_file in data_files:
        # To fix string formatting issues
        categorical_cols = data_file.select_dtypes(include=['object', 'str']).columns
        for col in categorical_cols:
            # The values are converted to strings, white spaces are removed and the strings are converted to lowercase
            data_file[col] = data_file[col].astype(str).str.strip().str.lower()

        numeric_cols = data_file.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Filling blank cells with median value of the columns
            data_file[col] = data_file[col].fillna(data_file[col].median())

    # To target outliers in the five data files
    outliers = {
        'players': (df_players, ['age', 'matches_played_2025_26', 'goals_2025_26']),
        'matches': (df_matches, ['predicted_win_prob_home', 'goals_home', 'goals_away', 'xG_home', 'xG_away']),
        'venues': (df_venues, ['capacity', 'pitch_quality_score']),
        'teams': (df_teams, ['fifa_rank', 'elo_rating', 'avg_age', 'goals_scored_last_12m', 'goals_conceded_last_12m']),
        'external_factors': (df_externalfactors, ['temperature', 'avg_age', 'travel_distance_km', 'fatigue_diff']),
    }

    # Used to round off the numbers
    for key, (data_file, cols) in outliers.items():
        for col in cols:
            if col in data_file.columns:
                lower_limit = data_file[col].quantile(0.01)
                upper_limit = data_file[col].quantile(0.99)
                data_file[col] = data_file[col].clip(lower_limit, upper_limit)
                
    print("Data successfully cleaned.")

    # Merging df_matches with df_teams twice (once each for both teams)
    df_merged = df_matches.merge(df_teams, left_on='team_home', right_on='team_id', how='left', suffixes=('', '_home'))
    df_merged = df_merged.merge(df_teams, left_on='team_away', right_on='team_id', how='left', suffixes=('_home', '_away'))
    
    # Merging df_venue and df_external factors values together
    df_merged = df_merged.merge(df_venues, on='venue_id', how='left')
    df_merged = df_merged.merge(df_externalfactors, on='match_id', how='left')

    # Used to store the outcome of the match, based on teams' scores
    df_merged['match_outcome'] = (df_merged['goals_home'] > df_merged['goals_away']).astype(int)


    safe_features = [
        'temperature', 'travel_distance_km', 'social_sentiment_home', 'social_sentiment_away',
        'referee_strictness', 'fatigue_diff', 'fifa_rank', 'elo_rating', 'squad_size', 
        'avg_age', 'market_value_million', 'goals_scored_last_12m', 'goals_conceded_last_12m',
        'fifa_rank_away', 'elo_rating_away', 'squad_size_away', 'avg_age_away', 
        'market_value_million_away', 'goals_scored_last_12m_away', 'goals_conceded_last_12m_away',
        'capacity', 'avg_temperature', 'humidity', 'pitch_quality_score', 'weather_condition'
    ]
    
    # Separating the final input features and target class array
    X = df_merged[[col for col in safe_features if col in df_merged.columns]]
    y = df_merged['match_outcome']

    # Used to convert text strings to numerical fields
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

    # Splitting rows for training the model and for verification
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Standardize numerical scales across features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if X_train is None:
        raise NotImplementedError

    print(f"Train size: {len(X_train_scaled)} | Test size: {len(X_test_scaled)}")

    # Starting the Logistic Regression algorithm
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    if model is None:
        raise NotImplementedError

    # Used for predicting outcomes on unseen samples
    predictions = model.predict(X_test_scaled)

    if predictions is None:
        raise NotImplementedError

    # Calculating evaluation scores
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    confusionMatrix = confusion_matrix(y_test, predictions)

    print(f"Logistic Regression Accuracy:  {accuracy:.4f}")
    print(f"Logistic Regression Precision: {precision:.4f}")
    print(f"Logistic Regression Recall:    {recall:.4f}")
    print("Confusion Matrix:")
    print(confusionMatrix)

    # Creating model paths safely
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "my_match_model.joblib")

    # Used to serialize and save trained model to disk
    joblib.dump(model, save_path)
    loaded_model = joblib.load(save_path)

    sample_match = X_test_scaled[[0]]
    
    predicted_outcome = loaded_model.predict(sample_match)[0]
    actual_outcome = y_test.iloc[0]
    print(f"Predicted Outcome: {predicted_outcome} (Actual Outcome: {actual_outcome})")

    print("The Machine Learning cycle has run successfully.")

if __name__ == "__main__":
    main()