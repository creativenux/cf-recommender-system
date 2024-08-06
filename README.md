# CF Recommender System

## Description
This project is a content-based filtering recommender system designed for university students. It provides personalized recommendations for books, societies, sports, and volunteer programs based on a student's country of origin and their academic and extracurricular interests.

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
- Uses content-based filtering and TF-IDF vectorization for accurate recommendations
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
2. Create a virtual environment: `python3 -m venv pyenv`
3. Activate the virtual environment: `source pyenv/bin/activate`
4. Install the required packages: `pip3 install -r requirements.txt`
5. Run the server: `fastapi dev server.py`
6. Open the `frontend` directory
7. Open the `index.html` file in your browser or start with live server if you have the extension installed in VS code

## Data Sources
The system uses the following datasets:
- Books: 4775 rows × 7 columns
- Country Sports: 200 rows × 2 columns
- Societies: 38 rows × 5 columns
- Sports: 32 rows × 3 columns
- Volunteer Programs: 55 rows × 4 columns

## Evaluation
- The system includes `evaluate_recommendations` method that assesses the performance of the recommender system using precision, recall, and F1 score metrics.
- The performance for books, societies, and volunteer programs is quite low, while the performance for sports is perfect.

Improvements suggestion

1. Sports (Perfect performance): The perfect score here because we are directly using the country's preferred sports as recommendations. While this ensures accuracy, it might limit variety.
2. Books (Very low performance):
   1. Expand the matching criteria like book summaries, keywords, or tags in addition to classification levels.
3. Societies (Low to moderate performance)
4. Volunteer Programs (Very low performance)