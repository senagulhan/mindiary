# 🧠 Mindiary

### *Machine Learning-Based Emotion Detection & Dynamic Wellness Recommendation System*

<p align="center">
  A wellness journaling application that detects emotions from journal entries and provides personalized recommendations using Machine Learning and Natural Language Processing.
</p>

---

## 🌐 Live Demo

🔗 **Try Mindiary:**
https://mindiary-7xe8.onrender.com/

📂 **GitHub Repository:**
https://github.com/senagulhan/mindiary

---

## 📖 About The Project

**Mindiary** is a machine learning-powered wellness journaling application developed as a **Bachelor's Graduation Thesis in Computer Engineering**.

Unlike traditional digital journals that simply store text, Mindiary analyzes journal entries, identifies emotional patterns, and transforms written thoughts into meaningful insights and personalized wellness suggestions.

The goal is to help users become more aware of their emotions and encourage healthy self-reflection through technology.

---

## ✨ Key Features

### 🎭 Emotion Detection

Analyze journal entries and detect emotions using NLP and Machine Learning.

Supported emotions:

* 😊 Joy
* 😢 Sadness
* 😠 Anger
* 😨 Fear
* 😲 Surprise
* ❤️ Love
* 😐 Neutral

The system predicts multiple emotions simultaneously and displays the **Top 3 dominant emotions** with confidence scores.

---

### 🤖 Machine Learning Pipeline

The emotion analysis process includes:

* Text preprocessing
* Contraction expansion
* Data cleaning
* TF-IDF Vectorization
* Logistic Regression Classification
* Multi-label emotion prediction
* Probability-based scoring

This allows the application to recognize complex emotional states instead of assigning only one emotion to an entire journal entry.

---

### 🧩 Progressive Chunking & Recency Bias

Mindiary doesn't evaluate a journal entry as a single block of text.

Instead, it:

* Analyzes different sections separately
* Tracks emotional changes throughout the entry
* Gives more weight to recent sentences

This approach better reflects how emotions often evolve while writing.

---

### 🌱 Personalized Wellness Suggestions

Based on detected emotions, Mindiary generates recommendations across multiple wellness categories:

🍲 Food

🏃 Activities

🎵 Music

📍 Places

🎨 Colors

🎬 Movies & TV Shows

👥 Social Suggestions

📚 Personal Growth Tasks

---

### ⭐ Adaptive Recommendation System

Users can:

* Mark recommendations as completed
* Leave ratings
* Provide feedback

The system stores this information and uses previous positive experiences to improve future recommendation rankings.

This creates a lightweight personalization mechanism inspired by retrieval-based recommendation systems.

---

## 📊 Wellness Chronicle Dashboard

Visualize emotional patterns over time.

### Available Filters

* 📅 Daily
* 📆 Weekly
* 🗓️ Monthly
* 📈 All Time

### Dashboard Insights

* Total Journal Entries
* Writing Streaks
* Dominant Mood
* Emotion Distribution
* Mood Trends
* Previous Entries
* Highest Rated Recommendations

The dashboard helps users understand long-term emotional behavior instead of focusing on a single journal entry.

---

## ⏰ Wellness Schedule & Email Reminders

Stay consistent with your journaling habit.

Features include:

* Daily reminder scheduling
* Weekly reflection generation
* Email notifications
* Habit-building support

When the configured reminder time arrives, Mindiary automatically sends reminder emails using the Brevo API.

---

## 🏗️ Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Flask

### Database

* PostgreSQL

### Machine Learning

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Logistic Regression

### APIs

#### 🎬 TMDB API

Used for:

* Movie recommendations
* TV Series recommendations

#### 🍽️ Spoonacular API

Used for:

* Food suggestions
* Recipe recommendations

#### 📧 Brevo API

Used for:

* Reminder emails
* Notification services

---

## ⚙️ How It Works

```text
User Journal Entry
        ↓
 Text Preprocessing
        ↓
 TF-IDF Vectorization
        ↓
 Emotion Classification
        ↓
 Top Emotion Detection
        ↓
 Wellness Recommendation Engine
        ↓
 User Feedback Collection
        ↓
 Personalized Future Recommendations
```

---

## 🔒 Security

User privacy is one of the project's main priorities.

Security measures include:

✅ Password Hashing

✅ Secure Authentication

✅ PostgreSQL Data Storage

✅ Protected User Information

✅ Separation of Credentials and Journal Content

Mindiary never stores passwords in plain text.

---

## 🎯 Project Goal

Many people struggle to identify and understand their emotions.

Mindiary aims to support emotional awareness by combining:

* 🧠 Machine Learning
* 💬 Natural Language Processing
* 🎯 Recommendation Systems
* 📈 Behavioral Feedback

into a single journaling experience.

The application encourages users to move from simply writing about emotions to actively understanding and responding to them.

---

## ⚠️ Disclaimer

Mindiary is **not a therapy platform, medical tool, or clinical diagnosis system.**

The application is designed solely to support:

* Emotional awareness
* Self-reflection
* Wellness journaling
* Personal growth

---

<p align="center">
  💙 Built with Machine Learning, NLP, and a passion for emotional well-being.
</p>
