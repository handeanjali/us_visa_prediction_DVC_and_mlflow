from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from us_visa.pipeline.prediction_pipeline import USvisaData, USvisaClassifier
from us_visa.pipeline.training_pipeline import TrainPipeline
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return render_template('usvisa.html', context="Rendering")


@app.route('/train', methods=['GET'])
def train_route():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return jsonify({"message": "Training successful!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['POST'])
def predict_route():
    try:
        form = request.form

        usvisa_data = USvisaData(
            continent=form.get("continent"),
            education_of_employee=form.get("education_of_employee"),
            has_job_experience=form.get("has_job_experience"),
            requires_job_training=form.get("requires_job_training"),
            no_of_employees=form.get("no_of_employees"),
            company_age=form.get("company_age"),
            region_of_employment=form.get("region_of_employment"),
            prevailing_wage=form.get("prevailing_wage"),
            unit_of_wage=form.get("unit_of_wage"),
            full_time_position=form.get("full_time_position"),
        )

        usvisa_df = usvisa_data.get_usvisa_input_data_frame()
        model_predictor = USvisaClassifier()
        prediction = model_predictor.predict(dataframe=usvisa_df)[0]

        status = "Visa-approved" if prediction == 1 else "Visa Not-Approved"
        return render_template('usvisa.html', context=status)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
