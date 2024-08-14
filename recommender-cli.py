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
    
    print("Running model evaluation...")
    evaluation_results = recommender.evaluate_recommendations()
    [ avg_scores, (overall_avg_precision, overall_avg_recall, overall_avg_f1) ] = evaluation_results
    print("\nOverall Results:")
    print(f"Overall Average Precision: {overall_avg_precision:.4f}")
    print(f"Overall Average Recall: {overall_avg_recall:.4f}")
    print(f"Overall Average F1-score: {overall_avg_f1:.4f}")
 
if __name__ == "__main__":
    main()