# Course Recommendation System

## Overview
The Course Recommendation System is a web application that uses Natural Language Processing (NLP) and machine learning techniques to recommend courses based on user input. The project processes a dataset of courses from Coursera and leverages TF-IDF vectorization with cosine similarity to suggest similar courses.

## Features
- **Course Recommendations:** Get similar courses based on a given course name.
- **NLP & Machine Learning:** Utilizes TF-IDF vectorization, cosine similarity, and stemming for text processing.
- **Responsive Web Interface:** Built with Flask and Bootstrap for a modern, mobile-friendly design.
- **Dynamic Interaction:** AJAX-powered form submission for real-time recommendations.

## Project Structure
- **`main.py`**  
  Processes the dataset (from `Coursera.csv`), cleans and preprocesses text data, calculates similarity scores, and exports the trained model to `course_recommender_model.pkl`.

- **`app.py`**  
  A Flask application that loads the exported model and handles user requests for course recommendations. It renders the main interface and provides recommendations via AJAX calls.

- **`index.html`**  
  The HTML template that defines the user interface. It uses Bootstrap for styling and includes a datalist for course name suggestions.

- **`style.css`**  
  Custom CSS to further enhance the user interface with a modern look and feel.

- **`Coursera.csv`**  
  The dataset containing course information. This file should be placed in the project root or an appropriate data folder.

## Getting Started

### Prerequisites
- **Python 3.x** installed on your machine.
- It is recommended to use a virtual environment.
- Required Python packages:
  - Flask
  - Pandas
  - Scikit-Learn
  - NLTK

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sri-venkat-22/Course-Recommendation.git
