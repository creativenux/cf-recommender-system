# import required packages
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import re

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
            recommendations['sports'].extend(preferred_sports)

        # Get book recommendations based on academic interests
        for interest in academic_interests:
            mask_book = (
                self.books_df['classification_level_1'].str.contains(re.escape(interest), case=False, na=False, regex=True)
                |
                self.books_df['classification_level_2'].str.contains(re.escape(interest), case=False, na=False,  regex=True)
                |
                self.books_df['classification_level_3'].str.contains(re.escape(interest), case=False, na=False,  regex=True)
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

    def evaluate_recommendations(self, num_iterations=100):
        results = {
            'books': [],
            'societies': [],
            'sports': [],
            'volunteer_programs': []
        }

        for _ in range(num_iterations):
            # Randomly select a country
            country = np.random.choice(self._get_countries_list())

            # Randomly select academic interests
            academic_interests = np.random.choice(
                self._get_academic_activities_list(),
                size=min(4, len(self._get_academic_activities_list())),
                replace=False
            )

            # Randomly select extracurricular interests
            extracurricular_interests = np.random.choice(
                self._get_extracurricular_activities_list(),
                size=min(4, len(self._get_extracurricular_activities_list())),
                replace=False
            )

            # Get recommendations
            recommendations = self.get_recommendations(country, academic_interests, extracurricular_interests)

            # Evaluate each category
            for category in results.keys():
                if category == 'sports':
                    # For sports, check if recommended sports are in the country's preferred sports
                    preferred_sports = self.country_sports_df[self.country_sports_df['country'] == country]['preferred_sport'].values[0].split(', ')
                    relevant = set(recommendations[category]) & set(preferred_sports)
                elif category == 'books':
                    # For books, check if recommended books match any of the academic interests
                    relevant = [book for book in recommendations[category] if any(interest.lower() in book.lower() for interest in academic_interests)]
                else:
                    # For societies and volunteer programs, check if recommended items match any of the extracurricular interests
                    relevant = [item for item in recommendations[category] if any(interest.lower() in item.lower() for interest in extracurricular_interests)]

                precision = len(relevant) / len(recommendations[category]) if recommendations[category] else 0
                recall = len(relevant) / len(set(preferred_sports if category == 'sports' else (academic_interests if category == 'books' else extracurricular_interests)))
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                results[category].append({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

        # Calculate average scores
        avg_scores = {}
        for category, scores in results.items():
            avg_scores[category] = {
                'avg_precision': np.mean([s['precision'] for s in scores]),
                'avg_recall': np.mean([s['recall'] for s in scores]),
                'avg_f1': np.mean([s['f1'] for s in scores])
            }

        # Print results
        # for category, scores in avg_scores.items():
        #     print(f"\nResults for {category}:")
        #     print(f"Average Precision: {scores['avg_precision']:.4f}")
        #     print(f"Average Recall: {scores['avg_recall']:.4f}")
        #     print(f"Average F1-score: {scores['avg_f1']:.4f}")

        # Calculate overall average scores
        overall_avg_precision = np.mean([scores['avg_precision'] for scores in avg_scores.values()])
        overall_avg_recall = np.mean([scores['avg_recall'] for scores in avg_scores.values()])
        overall_avg_f1 = np.mean([scores['avg_f1'] for scores in avg_scores.values()])

        # print("\nOverall Results:")
        # print(f"Overall Average Precision: {overall_avg_precision:.4f}")
        # print(f"Overall Average Recall: {overall_avg_recall:.4f}")
        # print(f"Overall Average F1-score: {overall_avg_f1:.4f}")

        return [avg_scores, (overall_avg_precision, overall_avg_recall, overall_avg_f1)]
    
    def get_links(self, recommendations):
        links = {
            'societies': self.societies_df[self.societies_df['society_name'].isin(recommendations['societies'])]['society_link'].tolist(),
            'sports': self.sports_df[self.sports_df['sport_name'].isin(recommendations['sports'])]['sport_link'].tolist(),
            'volunteer_programs': self.volunteer_programs_df[self.volunteer_programs_df['program_name'].isin(recommendations['volunteer_programs'])]['program_link'].tolist()
        }
        return links