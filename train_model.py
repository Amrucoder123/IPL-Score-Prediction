import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load data
ipl_df = pd.read_csv('ipl_data.csv')

# Remove irrelevant columns
ipl_df = ipl_df.drop(['mid', 'date', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'], axis=1)

# Keep only consistent teams
const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
              'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
              'Delhi Daredevils', 'Sunrisers Hyderabad']
ipl_df = ipl_df[(ipl_df['bat_team'].isin(const_teams)) & (ipl_df['bowl_team'].isin(const_teams))]

# Remove first 5 overs
ipl_df = ipl_df[ipl_df['overs'] >= 5.0]

# Label encoding
le = LabelEncoder()
for col in ['bat_team', 'bowl_team']:
    ipl_df[col] = le.fit_transform(ipl_df[col])

# One-hot encoding
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
ipl_df = np.array(columnTransformer.fit_transform(ipl_df))

# Create DataFrame with proper columns
cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
        'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
        'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
        'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
        'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
        'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
        'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(ipl_df, columns=cols)

# Prepare features and labels
features = df.drop(['total'], axis=1)
labels = df['total']

# Split data
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.20, shuffle=True, random_state=42)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_features, train_labels)

# Evaluate
train_score = model.score(train_features, train_labels) * 100
test_score = model.score(test_features, test_labels) * 100
print(f'Train Score: {train_score:.2f}%')
print(f'Test Score: {test_score:.2f}%')

# Save model
with open('ml_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'ml_model.pkl'")
print("You can now run: streamlit run ipl_score_predictor.py")