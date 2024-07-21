from fastapi import FastAPI
from recommender import RecommenderSystem

# create a new instance of our recommender
recommenderSystem = RecommenderSystem()

app = FastAPI()

@app.get("/")
def index():
    return {"msg": "Hello world"}


@app.get("/user-options")
def get_user_options():
    user_options = recommenderSystem.get_user_options()
    return {
        'data': user_options
    }

@app.post("/recommend")
def recommend(user_options: dict):
    country, academic_interests, extracurricular_interests = recommenderSystem.compile_user_options({
            'country': user_options['country'],
            'academic_interests': user_options['academic_interests'],
            'extracurricular_interests': user_options['extracurricular_interests']
        })
    
    recommended_items = recommenderSystem.get_recommendations(country, academic_interests, extracurricular_interests)
    return {
        'data': recommended_items
    }
