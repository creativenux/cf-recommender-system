from fastapi import FastAPI
from recommender import RecommenderSystem

# create a new instance of our recommender
recommenderSystem = RecommenderSystem()

app = FastAPI()

@app.get("/user-options")
def get_user_options():
    user_options = recommenderSystem.get_user_options()
    return user_options
