
import os
import flask
import pickle
import time

app = flask.Flask(__name__)

production_folder = os.path.join(os.getcwd(), 'production')
artifacts_folder = os.path.join(production_folder, 'artifacts')
cleaning_pipeline_name = 'cleaning_pipeline.pickle'
model_pipeline_name = 'model.pickle'
cleaning_pipeline_path = os.path.join(artifacts_folder, cleaning_pipeline_name)
model_pipeline_path = os.path.join(artifacts_folder, model_pipeline_name)

with open(cleaning_pipeline_path, 'rb') as file:
    cleaning_pipeline = pickle.load(file)

with open(model_pipeline_path, 'rb') as file:
    model_pipeline = pickle.load(file)


@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        time_start = time.time()
        data = flask.request.get_json(force=True)
        text = data['twitter']
        text = [text]

        text_cleaned = cleaning_pipeline.transform(text)
        prediction = model_pipeline.predict(text_cleaned)[0]

        time_end = time.time() - time_start
        result = f'Tag: {prediction}, inference time: {time_end:.4f} seconds'

        return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False, port=8000)
