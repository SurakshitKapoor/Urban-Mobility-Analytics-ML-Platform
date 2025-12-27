


from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask app is running!"


if __name__ == "__main__":
    print("app.py is starting point!")
    # app.run(debug=True)
    app.run()