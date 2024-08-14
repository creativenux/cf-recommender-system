from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

app.mount("/static", StaticFiles(directory="frontend"), name="frontend")
templates = Jinja2Templates(directory="frontend")

@app.get("/")
async def index(request: Request, id: str = None):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"id": id}
    )


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
    
    no_of_recommendation = int(user_options['no_of_recommendation']) or 3
    recommended_items = recommenderSystem.get_recommendations(country, academic_interests, extracurricular_interests, no_of_recommendation)
    return {
        'data': recommended_items
    }

@app.get("/evaluation")
def get_performance_evaluation():
    evaluation_results = recommenderSystem.evaluate_recommendations()
    [ avg_scores, (overall_avg_precision, overall_avg_recall, overall_avg_f1) ] = evaluation_results
    return {
        'data': {
            'avg_scores': avg_scores,
            'overall_avg_precision': overall_avg_precision,
            'overall_avg_recall': overall_avg_recall,
            'overall_avg_f1': overall_avg_f1
        }
    }