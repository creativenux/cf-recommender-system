from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recommender import RecommenderSystem

# create a new instance of our recommender
recommenderSystem = RecommenderSystem()

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            'country': int(user_options['country']),
            'academic_interests': user_options['academic_interests'],
            'extracurricular_interests': user_options['extracurricular_interests']
        })
    
    print(user_options)

    no_of_recommendation = int(user_options['no_of_recommendation']) or 3
    recommended_items = recommenderSystem.get_recommendations(country, academic_interests, extracurricular_interests, no_of_recommendation)
    return {
        'data': recommended_items
    }
