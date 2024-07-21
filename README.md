# CF Recommender System

## Description
This project is a collaborative filtering recommender system designed for university students. It provides personalized recommendations for books, societies, sports, and volunteer programs based on a student's country of origin and their academic and extracurricular interests.

## Features
- Personalized recommendations for:
  - Books
  - Societies
  - Sports
  - Volunteer programs
- Based on:
  - Country of origin
  - Academic interests
  - Extracurricular interests
- Uses collaborative filtering and TF-IDF vectorization for accurate recommendations
- Includes an evaluation method to assess the recommender system's performance **(Not working yet)**

## Installation
1. Clone the repository: `git clone https://github.com/creativenux/cf-recommender-system.git`
2. Navigate to the project directory: `cd cf-recommender-system`

## Run on jupyter notebook
1. Open the project with juptyer notebook
2. Navigate to `notebook` directory
3. There are 2 main scrips you can run: 
   1. `recommender-class` - This is an average system
   2. `recommender-main` - This provides better recommendation compare to the first
4. Follow the prompts to input your preferences and receive personalized recommendations.

## Run on terminal
1. Open the terminal and navigate to the project directory
2. Install the required packages: `pip3 install -r requirements.txt` 
3. Run the recommender system: `python3 main.py`

## Run the server and the webapp
1. Open the terminal and navigate to the project directory
2. Install the required packages: `pip3 install -r requirements.txt`
3. Run the server: `fastapi dev server.py`
4. Open the `frontend` directory
5. Open the `index.html` file in your browser or start with live server

## Data Sources
The system uses the following datasets:
- Books: 4775 rows × 7 columns
- Country Sports: 200 rows × 2 columns
- Societies: 38 rows × 5 columns
- Sports: 32 rows × 3 columns
- Volunteer Programs: 55 rows × 4 columns

## Evaluation (Not working yet)
The system includes an `evaluate_recommender` method that assesses the performance of the recommender system using precision, recall, and F1 score metrics.

