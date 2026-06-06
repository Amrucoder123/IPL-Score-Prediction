# IPL Score Predictor

This Streamlit webapp enables user to predict total runs between teams using current runs and wickets.

**Algorithms used:**

* Linear Regression
* K-Nearest Neighbor Regressor 
* XGBoost Regressor
* RandomForest Regressor
* SVR
* Decision Tree Regressor

**Hyperparamter Optimization:**

Used optuna for paramter optimization.

**Dataset:**

**Dataset:** IPL ball-by-ball data (2008-2017 originally; model retrained on same for consistency with legacy team names: KKR, CSK, RR, MI, KXIP, RCB, DD, SRH). Source: Public IPL dataset.

Dataset Used: ipl_data.csv

* mid - match id
* date - when matches are played
* venue - place where matches aew played
* bat_team - batting team
* bowl_team - bowling team
* batsman - batsman
* bowler - bowler
* runs - runs scored
* wickets - wickets
* overs - overs - next 3 are based on this
* run_last_5 - runs scored in last 5 overs
* wicket_last_5 - wickets in last 5 overs
* stricker - batsman playing as main 1
* non-striker - batsman playing as runner up - not main 0
* total - total score (target variable)

## How to Run

### Prerequisites
```bash
python 3.7+
```

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model (if ml_model.pkl doesn't exist):**
```bash
jupyter notebook IPL_Score_Predictor.ipynb
```
Run all cells to generate `ml_model.pkl`

3. **Run the Streamlit app:**
```bash
streamlit run ipl_score_predictor.py
```

The app will open in your browser at `http://localhost:8501`

**Streamlit App:**
python -m streamlit run ipl_score_predictor.py

train model 

python train_model.py