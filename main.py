from recommender import RecommenderSystem

def main():
    # start our recommender
    recommender = RecommenderSystem()
    country, academic_interests, extracurricular_interests, no_of_recommendations = recommender.ask_user_preferences()
    recommendations = recommender.get_recommendations(country, academic_interests, extracurricular_interests, no_of_recommendations)

    print("Here are some recommendations for you:")
    for category, items in recommendations.items():
        if items:
            print(f"{category.capitalize()}: {', '.join(items)}")
        else:
            print(f"{category.capitalize()}: No recommendations found")
 
if __name__ == "__main__":
    main()