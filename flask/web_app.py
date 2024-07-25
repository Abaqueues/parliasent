from flask import Flask, render_template, request
import psycopg2 as ps2

app = Flask(__name__)

_DATABASE_URI = "postgresql://postgres:password@127.0.0.1:5432/predicted_data"

@app.route("/")
@app.route("/index/")
def index():
    return render_template("index.html")

@app.route("/data/")
def data():
    conn = ps2.connect(_DATABASE_URI)
    cursor = conn.cursor()
    cursor.execute("SELECT index, id, speakername, url, date, sgd_prediction FROM speech_data")
    data = cursor.fetchall()
    cursor.close()
    return render_template("data.html", data = data) 

@app.route("/search/")
def search():
    return render_template("form.html")

@app.route("/process_query/", methods=["GET", "POST"])
def process_form():
    user_input = request.form.get('user_input')
    conn = ps2.connect(_DATABASE_URI)
    cursor = conn.cursor()
    cursor.execute(f"SELECT index, id, speakername, url, date, lr_prediction FROM speech_data WHERE speakername = %(user_input)s", {'user_input': user_input})
    data = cursor.fetchall()
    cursor.close()
    return render_template("query.html", data = data) 

# @app.teardown_appcontext
# def close_connection(exception):
#     conn.close()

if __name__ == "__main__":
    app.run()
    