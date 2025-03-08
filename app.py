from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load model
with open('course_recommender_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
similarity = model_data['similarity']
new_df = pd.DataFrame(model_data['courses'])

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def process_text(text):
    words = nltk.word_tokenize(text.lower())
    stemmed = [ps.stem(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(stemmed)

def recommend(course, num_recommendations=6):
    try:
        course_matches = new_df[new_df['course_name'].str.contains(course, case=False)]
        if len(course_matches) == 0:
            return "Course not found. Please try another course name."
        course_index = course_matches.index[0]
        distances = similarity[course_index]
        course_indices = sorted(
            [(i, score) for i, score in enumerate(distances) if i != course_index],
            key=lambda x: x[1], reverse=True
        )[:num_recommendations]
        return [
            {'course_name': new_df.iloc[idx]['course_name'], 'similarity_score': round(score * 100, 2)}
            for idx, score in course_indices
        ]
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/', methods=['GET'])
def index():
    # Pass list of course names to populate suggestions
    return render_template('index.html', course_names=new_df['course_name'].tolist())

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    course = request.form.get('course', '')
    if not course:
        return jsonify({"error": "Please provide a course name"}), 400
    results = recommend(course)
    if isinstance(results, str):
        return jsonify({"error": results}), 400
    return jsonify({"recommendations": results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
