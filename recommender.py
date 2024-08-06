# import required packages
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class RecommenderSystem:
    def __init__(self):
        
        # Load datasets
        self.books_df = pd.read_csv('./dataset/books.csv')
        self.country_sports_df = pd.read_csv('./dataset/country_sports.csv')
        self.societies_df = pd.read_csv('./dataset/societies.csv')
        self.sports_df = pd.read_csv('./dataset/sports.csv')
        self.volunteer_programs_df = pd.read_csv('./dataset/volunteer_programs.csv')
        
        self._preprocess_data()
        self._create_similarity_matrices()
    
    
    def _preprocess_data(self):
        # Combine columns into a single string for TF-IDF, handling missing values
        self.books_df['combined'] = self.books_df[['book_title', 'book_author', 'classification_level_1', 'classification_level_2', 'classification_level_3']].fillna('').agg(' '.join, axis=1)
        self.societies_df['combined'] = self.societies_df[['society_name', 'society_type', 'society_description', 'society_keywords']].fillna('').agg(' '.join, axis=1)
        self.sports_df['combined'] = self.sports_df[['sport_name', 'sport_description', 'sport_keywords']].fillna('').agg(' '.join, axis=1)
        self.volunteer_programs_df['combined'] = self.volunteer_programs_df[['program_name', 'program_description', 'program_keywords']].fillna('').agg(' '.join, axis=1)
    
    def _create_similarity_matrices(self):
        # Apply TF-IDF Vectorizer
        tfidf_books = TfidfVectorizer(stop_words='english')
        self.cosine_sim_books = cosine_similarity(tfidf_books.fit_transform(self.books_df['combined']))
        
        tfidf_societies = TfidfVectorizer(stop_words='english')
        self.cosine_sim_societies = cosine_similarity(tfidf_societies.fit_transform(self.societies_df['combined']))
        
        tfidf_sports = TfidfVectorizer(stop_words='english')
        self.cosine_sim_sports = cosine_similarity(tfidf_sports.fit_transform(self.sports_df['combined']))
        
        tfidf_volunteer_programs = TfidfVectorizer(stop_words='english')
        self.cosine_sim_volunteer_programs = cosine_similarity(tfidf_volunteer_programs.fit_transform(self.volunteer_programs_df['combined']))

    def _get_recommendations(self, item_name, df, cosine_sim, num_recommendations=3):
        idx = df[df.iloc[:, 1] == item_name].index
        if len(idx) == 0:
            print(f"Warning: '{item_name}' not found in the dataset.")
            return pd.DataFrame()  # Return an empty DataFrame if item is not found

        idx = idx[0]  # Get the first index if found
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        item_indices = [i[0] for i in sim_scores]
        return df.iloc[item_indices].iloc[:, 1:3]

    def get_book_recommendations(self, book_title, num_recommendations=3):
        recommendations = self._get_recommendations(book_title, self.books_df, self.cosine_sim_books, num_recommendations)
        if recommendations.empty:
            return []
        return recommendations['book_title'].tolist()

    def get_society_recommendations(self, society_name, num_recommendations=3):
        return self._get_recommendations(society_name, self.societies_df, self.cosine_sim_societies, num_recommendations)

    def get_sport_recommendations(self, sport_name, num_recommendations=3):
        return self._get_recommendations(sport_name, self.sports_df, self.cosine_sim_sports, num_recommendations)

    def get_volunteer_program_recommendations(self, program_name, num_recommendations=3):
        return self._get_recommendations(program_name, self.volunteer_programs_df, self.cosine_sim_volunteer_programs, num_recommendations)

    def _get_countries_list(self):
        return self.country_sports_df['country'].unique()
    
    def _get_academic_activities_list(self):
        return sorted(self.books_df[['classification_level_1', 'classification_level_2', 'classification_level_3']].stack().reset_index(drop=True).unique().tolist())
    
    def _get_extracurricular_activities_list(self):
        extracurricular_keywords = set(
            self.societies_df['society_keywords'].str.split(', ').sum() +
            self.volunteer_programs_df['program_keywords'].str.split(', ').sum()
        )
        return list(extracurricular_keywords)
    
    def get_user_options(self):
        return {
            "countries": list(self._get_countries_list()), 
            "academic_activities": list(self._get_academic_activities_list()), 
            "extracurricular_activities": list(self._get_extracurricular_activities_list())
        }
    
    def compile_user_options(self, options):
        country_index = options['country']
        academic_indices = options['academic_interests']
        extracurricular_indices = options['extracurricular_interests']

        country = self._get_countries_list()[country_index-1]

        academic_indices = [int(i.strip()) - 1 for i in academic_indices.split(',')]
        academic_activities_list = self._get_academic_activities_list()
        academic_interests = [academic_activities_list[i] for i in academic_indices]

        extracurricular_indices = [int(i.strip()) - 1 for i in extracurricular_indices.split(',')]
        extracurricular_activities_list = self._get_extracurricular_activities_list()
        extracurricular_interests = [list(extracurricular_activities_list)[i] for i in extracurricular_indices]

        return country, academic_interests, extracurricular_interests

    def ask_user_preferences(self):
        # prompt use to select their country of origin
        print("Please select your country of origin:")
        countries = self._get_countries_list()
        for i, country in enumerate(countries, 1):
            print(f"{i}. {country}")

        country_index = int(input("Enter the number corresponding to your country: ")) - 1
        
        # create a list of academic activities based on book classification levels
        academic_activities = self._get_academic_activities_list()
        print("\nPlease select your top 3 academic interests:")
        for i, activity in enumerate(academic_activities, 1):
            print(f"{i}. {activity}")

        academic_indices = input("Enter the numbers of your top 3 academic interests, separated by commas: ")
        
        # create a list of extracurricular activities using keywords from society and program dataset
        extracurricular_keywords = self._get_extracurricular_activities_list()
        print("\nPlease select your top 3 extracurricular interests:")
        for i, keyword in enumerate(extracurricular_keywords, 1):
            print(f"{i}. {keyword}")

        extracurricular_indices = input("Enter the numbers of your top 3 extracurricular interests, separated by commas: ")
        
        country, academic_interests, extracurricular_interests = self.compile_user_options({
            'country': country_index,
            'academic_interests': academic_indices,
            'extracurricular_interests': extracurricular_indices
        })

        # prompt user to enter no of recommendations they want
        no_of_recommendations = int(input("Please enter the max no of recommendations you want (we recommend 3): ")) or 3

        return country, academic_interests, extracurricular_interests, no_of_recommendations

    def get_recommendations(self, country, academic_interests, extracurricular_interests, no_of_recommendation=3):

        recommendations = {
            'books': [],
            'societies': [],
            'sports': [],
            'volunteer_programs': []
        }

        # Get sports recommendations based on country
        preferred_sports = self.country_sports_df[self.country_sports_df['country'] == country]['preferred_sport'].values
        if len(preferred_sports) > 0:
            preferred_sports = preferred_sports[0].split(', ')
#             for sport in preferred_sports:
#                 sport_recs = self.get_sport_recommendations(sport)
#                 recommendations['sports'].extend(sport_recs['sport_name'].tolist())
            # set the preferred_sports in the country-sports data set as the sport recommendation
            recommendations['sports'].extend(preferred_sports)

        # Get book recommendations based on academic interests
        for interest in academic_interests:
            mask_book = (
                self.books_df['classification_level_1'].str.contains(interest, case=False, na=False)
                |
                self.books_df['classification_level_2'].str.contains(interest, case=False, na=False)
                |
                self.books_df['classification_level_3'].str.contains(interest, case=False, na=False)
            )
            book_recs = self.books_df[mask_book]['book_title'].tolist()
            recommendations['books'].extend(book_recs)

        # Get society and volunteer program recommendations based on extracurricular interests
        for interest in extracurricular_interests:
            society_recs = self.societies_df[
                self.societies_df['society_keywords'].str.contains(interest, case=False, na=False) |
                self.societies_df['society_description'].str.contains(interest, case=False, na=False)
            ]['society_name'].tolist()
            recommendations['societies'].extend(society_recs)

            program_recs = self.volunteer_programs_df[
                self.volunteer_programs_df['program_keywords'].str.contains(interest, case=False, na=False) |
                self.volunteer_programs_df['program_description'].str.contains(interest, case=False, na=False)
            ]['program_name'].tolist()
            recommendations['volunteer_programs'].extend(program_recs)

        # Ensure maximum of 3 unique recommendations for each category
        for category in recommendations:
            recommendations[category] = list(set(recommendations[category]))[:no_of_recommendation]

        return recommendations

    # TODO: fix book recommendation bug here
    def evaluate_recommender(self, num_iterations=100):
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for _ in range(num_iterations):
            # Split the data into training and testing sets
            train_books, test_books = train_test_split(self.books_df, test_size=0.2)
            train_societies, test_societies = train_test_split(self.societies_df, test_size=0.2)
            train_sports, test_sports = train_test_split(self.sports_df, test_size=0.2)
            train_volunteer, test_volunteer = train_test_split(self.volunteer_programs_df, test_size=0.2)

            # Create temporary recommendation system with training data
            temp_system = RecommenderSystem()
            temp_system.books_df = train_books
            temp_system.societies_df = train_societies
            temp_system.sports_df = train_sports
            temp_system.volunteer_programs_df = train_volunteer
            temp_system._preprocess_data()
            temp_system._create_similarity_matrices()

            # Generate recommendations for test data
            true_labels = []
            predicted_labels = []

            def process_recommendations(recs, train_data, id_column):
                if recs.empty:
                    return
                true_labels.extend([1] + [0] * (len(recs) - 1))
                predicted_labels.extend([1 if item in train_data[id_column].values else 0 for item in recs[id_column]])

            for _, row in test_books.iterrows():
                recs = temp_system.get_book_recommendations(row['book_title'])
                process_recommendations(recs, train_books, 'book_title')

            for _, row in test_societies.iterrows():
                recs = temp_system.get_society_recommendations(row['society_name'])
                process_recommendations(recs, train_societies, 'society_name')

            for _, row in test_sports.iterrows():
                recs = temp_system.get_sport_recommendations(row['sport_name'])
                process_recommendations(recs, train_sports, 'sport_name')

            for _, row in test_volunteer.iterrows():
                recs = temp_system.get_volunteer_program_recommendations(row['program_name'])
                process_recommendations(recs, train_volunteer, 'program_name')

            # Calculate metrics only if we have predictions
            if predicted_labels:
                precision_scores.append(precision_score(true_labels, predicted_labels))
                recall_scores.append(recall_score(true_labels, predicted_labels))
                f1_scores.append(f1_score(true_labels, predicted_labels))

        # Return average scores
        if precision_scores:
            return {
                'precision': np.mean(precision_scores),
                'recall': np.mean(recall_scores),
                'f1': np.mean(f1_scores)
            }
        else:
            print("Warning: No valid recommendations were generated during evaluation.")
            return {
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
