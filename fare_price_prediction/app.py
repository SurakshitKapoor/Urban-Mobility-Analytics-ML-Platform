

from flask import Flask, render_template, request
import pandas as pd

from src.mlproject.pipelines.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            data = CustomData(
                passenger_type=request.form["passenger_type"],
                distance_km=float(request.form["distance_km"]),
                city_name=request.form["city_name"],
                day_category=request.form["day_category"],
                day=int(request.form["day"]),
                weekday=int(request.form["weekday"]),
                week=int(request.form["week"]),
                month_num=int(request.form["month_num"])
            )

            input_df = data.get_data_as_dataframe()

            predictor = PredictPipeline()
            prediction = predictor.predict(input_df)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)


# @app.route('/')
# def home():
#     return "<h1> this is Home Page! </h1>"


if __name__ == "__main__":
    print("starting from app.py")
    print("App running at: http://127.0.0.1:5000/")
    app.run(    debug=True    )







