from flask import Flask
from flask_restx import Resource, Api
from todo import Todo

app = Flask(__name__)
api = Api(app)

api.add_namespace(Todo, '/todos')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=81)
