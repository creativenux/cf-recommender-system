from recommender import RecommenderSystem

def main():
    # start our recommender
    recommender = RecommenderSystem()
    recommendations = recommender.get_recommendations()

    print("Here are some recommendations for you:")
    for category, items in recommendations.items():
        if items:
            print(f"{category.capitalize()}: {', '.join(items)}")
        else:
            print(f"{category.capitalize()}: No recommendations found")
 
if __name__ == "__main__":
    main()