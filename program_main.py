from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import os
import json
import re
import random
import datetime
import contractions
import requests
import urllib.parse
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()

try:
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import pickle
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn not found — keyword fallback only.")

app = Flask(__name__)
CORS(app,
     resources={r"/*": {"origins": "*"}},
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept"],
     supports_credentials=False
)

@app.before_request
def handle_preflight():
    from flask import request as req
    if req.method == "OPTIONS":
        from flask import Response
        resp = Response()
        resp.headers["Access-Control-Allow-Origin"]  = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
        resp.headers["Access-Control-Max-Age"]       = "86400"
        return resp





EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love', 'neutral']

def load_wellness_data():
    data_paths = [
        'wellness_data.json',
        os.path.join(os.path.dirname(__file__), 'wellness_data.json'),
    ]
    for path in data_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                print(f"✅ Loaded wellness_data.json from {path}")
                return json.load(f)
    print("⚠️  wellness_data.json not found — using empty data")
    return {"emotions": {}}

WELLNESS_DATA = load_wellness_data()

def get_db():
    # Bu URL'i .env dosyandan çekeceğiz
    conn_url = os.getenv("DATABASE_URL")
    # RealDictCursor sayesinde fetchone() ve fetchall() sonuçları otomatik dictionary döner
    return psycopg2.connect(conn_url, cursor_factory=RealDictCursor)

def fetchone_dict(cursor):
    return cursor.fetchone()

def fetchall_dict(cursor):
    return cursor.fetchall()

def _table_exists(cur, name):
    cur.execute(
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_NAME = %s", (name,))
    return cur.fetchone()['count'] > 0

def _column_exists(cur, table, column):
    cur.execute(
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_NAME=%s AND COLUMN_NAME=%s", (table, column))
    return cur.fetchone()['count'] > 0

def init_db(reset=False):
    conn = get_db()
    cur  = conn.cursor()

    # --- 1. VERİTABANINI TAMAMEN SIFIRLAMA (WIPE) BÖLÜMÜ ---
    if reset:
        print("⚠️ DİKKAT: Veritabanı tamamen sıfırlanıyor...")
        cur.execute("""
            DROP TABLE IF EXISTS user_feedback CASCADE;
            DROP TABLE IF EXISTS surveys CASCADE;
            DROP TABLE IF EXISTS recommendations CASCADE;
            DROP TABLE IF EXISTS emotions CASCADE;
            DROP TABLE IF EXISTS entries CASCADE;
            DROP TABLE IF EXISTS users CASCADE;
        """)
        conn.commit()
        print("🗑️ Tüm tablolar temizlendi, veri ortamı sıfırlandı.")

    # --- 2. TABLOLARI YENİDEN OLUŞTURMA BÖLÜMÜ ---
    if not _table_exists(cur, 'users'):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         SERIAL PRIMARY KEY,
                username   VARCHAR(150) UNIQUE NOT NULL,
                pw_hash    VARCHAR(64)  NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
    if not _table_exists(cur, 'entries'):
        cur.execute("""
            CREATE TABLE entries (
                id            SERIAL PRIMARY KEY,
                user_id       INT           NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                page_number   INT           NOT NULL DEFAULT 1,
                entry_date    VARCHAR(30)   NOT NULL,
                location      VARCHAR(255),
                entry_text    TEXT          NOT NULL,
                wellness_note TEXT,
                mood_score    FLOAT         DEFAULT NULL,
                created_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
                updated_at    TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
            )
        """)

    if not _table_exists(cur, 'emotions'):
        cur.execute("""
            CREATE TABLE emotions (
                id       SERIAL PRIMARY KEY,
                entry_id INT          NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
                emotion  VARCHAR(50)  NOT NULL,
                score    FLOAT        NOT NULL
            )
        """)

    if not _table_exists(cur, 'recommendations'):
        cur.execute("""
            CREATE TABLE recommendations (
                id       SERIAL PRIMARY KEY,
                entry_id INT          NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
                category VARCHAR(50)  NOT NULL,
                item     TEXT         NOT NULL,
                why      TEXT         DEFAULT ''
            )
        """)

    if not _table_exists(cur, 'surveys'):
        cur.execute("""
            CREATE TABLE surveys (
                id             SERIAL PRIMARY KEY,
                user_id        INT          NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                entry_id       INT,
                survey_date    VARCHAR(30)  NOT NULL,
                mood_score     INT          NOT NULL CHECK (mood_score BETWEEN 1 AND 10),
                liked_items    TEXT         DEFAULT '[]',
                disliked_items TEXT         DEFAULT '[]',
                free_text      TEXT         DEFAULT '',
                created_at     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            )
        """)

    if not _table_exists(cur, 'user_feedback'):
        cur.execute("""
            CREATE TABLE user_feedback (
                id            SERIAL PRIMARY KEY,
                user_id       INT          NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                item_text     TEXT         NOT NULL,
                category      VARCHAR(50)  NOT NULL,
                did_it        BOOLEAN      NOT NULL,
                stars         INT          DEFAULT 0,
                feedback_text TEXT         DEFAULT '',
                emotion       VARCHAR(50),
                created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            )
        """)

    # ALTER TABLE kontrolleri
    if not _column_exists(cur, 'recommendations', 'why'):
        cur.execute("ALTER TABLE recommendations ADD COLUMN why TEXT DEFAULT ''")
    if not _column_exists(cur, 'entries', 'mood_score'):
        cur.execute("ALTER TABLE entries ADD COLUMN mood_score FLOAT DEFAULT NULL")
    if not _column_exists(cur, 'entries', 'card_feedback'):
        cur.execute("ALTER TABLE entries ADD COLUMN card_feedback TEXT DEFAULT '{}'")

    conn.commit()
    conn.close()
    print('✅ PostgreSQL database ready.')

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# TF-IDF + LOGISTIC REGRESSION MODEL

tfidf_vectorizer  = None
tfidf_classifiers = {}   # label → trained LogisticRegression

BEST_C = {
    'joy':      5.0,
    'sadness':  5.0,
    'anger':    5.0,
    'fear':     5.0,
    'surprise': 5.0,
    'love':     2.0,
    'neutral':  5.0,
}

STRONG_KEYWORDS = {
    'sadness': [
        # feel bad / negative state (kısa kalıplar)
        'feel bad', 'feel terrible', 'feel awful', 'feel horrible', 'feel sick',
        'feel off', 'feel down', 'feel low', 'feel blue',
        'feeling bad', 'feeling terrible', 'feeling awful', 'feeling horrible',
        'feeling down', 'feeling low', 'feeling blue', 'feeling sick',
        'felt bad', 'felt terrible', 'felt awful', 'felt horrible',
        # isolation / emptiness
        'feel lonely', 'feel empty', 'feel lost', 'feel broken', 'feel hollow',
        'feel numb', 'feel hopeless', 'feel worthless', 'feel useless',
        'feeling lonely', 'feeling empty', 'feeling lost', 'feeling broken',
        'feeling numb', 'feeling hopeless', 'feeling worthless',
        # exhaustion
        'feel tired', 'feel exhausted', 'feel drained', 'feel worn out',
        'feeling tired', 'feeling exhausted', 'feeling drained',
        # depression
        'feel depressed', 'feeling depressed', 'feel miserable', 'feeling miserable',
        # crying / grief
        'i cried', 'i am crying', 'i was crying', 'i keep crying',
        'miss her', 'miss him', 'miss them', 'miss you', 'i miss',
        'so sad', 'i am sad', 'i feel sad', 'feeling sad',
        'hurts so much', 'it hurts', 'my heart hurts',
        'i found myself crying', 'tears down', 'tears in my eyes',
        'not good enough', 'hate myself',
        
        # --- EKLENEN KÖK KELİMELER ---
        'heartbroken', 'devastated', 'miserable', 'depressed', 'lonely',
        'hopeless', 'worthless', 'grief', 'sorrow', 'crying', 'tears', 'suicidal'
    ],
    'anger': [
        'feel angry', 'feeling angry', 'feel mad', 'feeling mad',
        'feel furious', 'feeling furious', 'feel outraged',
        'so angry', 'so mad', 'so furious',
        'makes me angry', 'makes me mad', 'made me angry', 'made me mad',
        'i am angry', 'i am mad', 'i am furious', 'i was angry',
        'i hate', 'i despise', 'i cannot stand',
        'so frustrated', 'so irritated', 'so annoyed',
        'feel frustrated', 'feeling frustrated', 'feel irritated', 'feeling irritated',
        'drives me crazy', 'drives me mad', 'drives me nuts',
        'pissed off', 'fed up', 'sick of this', 'tired of this',
        'so unfair', 'not fair', 'i am outraged', 'makes me furious',
        
        # --- EKLENEN KÖK KELİMELER ---
        'furious', 'infuriating', 'blood boil', 'ruined', 'derailed',
        'frustrating', 'stupid', 'outraged', 'pissed', 'resentful', 'rage'
    ],
    'fear': [
        'feel scared', 'feeling scared', 'feel afraid', 'feeling afraid',
        'feel anxious', 'feeling anxious', 'feel nervous', 'feeling nervous',
        'feel worried', 'feeling worried', 'feel terrified', 'feeling terrified',
        'feel stressed', 'feeling stressed', 'feel overwhelmed', 'feeling overwhelmed',
        'feel panicked', 'feel panicky',
        'i am scared', 'i am afraid', 'i am anxious', 'i am nervous',
        'i am worried', 'i am terrified', 'i am stressed', 'i am overwhelmed',
        'so scared', 'so afraid', 'so anxious', 'so nervous', 'so worried',
        'so stressed', 'so overwhelmed', 'so terrified',
        'heart is pounding', 'heart is racing', 'cannot breathe',
        'panic attack', 'freaking out', 'on edge', 'i dread',
        'i am petrified', 'filled with dread', 'losing my mind',
        
        # --- EKLENEN KÖK KELİMELER ---
        'terrified', 'anxious', 'panic', 'dread', 'petrified', 'scared',
        'afraid', 'nervous', 'overwhelmed', 'nightmare', 'spiraling'
    ],
    'joy': [
        'feel good', 'feeling good', 'feel great', 'feeling great',
        'feel happy', 'feeling happy', 'feel wonderful', 'feeling wonderful',
        'feel amazing', 'feeling amazing', 'feel fantastic', 'feeling fantastic',
        'feel excited', 'feeling excited', 'feel thrilled', 'feeling thrilled',
        'feel blessed', 'feeling blessed', 'feel grateful', 'feeling grateful',
        'feel content', 'feeling content', 'feel joyful', 'feeling joyful',
        'feel elated', 'feeling elated', 'feel overjoyed',
        'so happy', 'so excited', 'so grateful', 'so blessed', 'so thrilled',
        'i am happy', 'i am excited', 'i am grateful', 'i am thrilled',
        'best day', 'great day', 'wonderful day', 'amazing day',
        'over the moon', 'on top of the world', 'could not be happier',
        'made me smile', 'made me laugh', 'cheered me up',
        
        # --- EKLENEN KÖK KELİMELER ---
        'thrilled', 'ecstatic', 'overjoyed', 'amazing', 'fantastic',
        'grateful', 'blessed', 'delighted', 'joyful', 'cheerful', 'awesome'
    ],
    'love': [
        'i love you', 'love you', 'i adore', 'i cherish',
        'head over heels', 'in love with', 'love of my life',
        'miss you so much', 'i adore her', 'i adore him',
        'feel love', 'feeling love', 'feel loved', 'feeling loved',
        'so much love', 'full of love', 'so in love',
        'my heart belongs', 'you mean everything', 'everything to me',
        
        # --- EKLENEN KÖK KELİMELER ---
        'adore', 'cherish', 'affection', 'devoted', 'beloved', 'soulmate'
    ],
    'surprise': [
        'cannot believe', 'can not believe', 'could not believe',
        'i was shocked', 'i am shocked', 'totally shocked', 'completely shocked',
        'out of nowhere', 'did not expect', 'never expected',
        'blown away', 'mind blown', 'caught off guard',
        'completely unexpected', 'never saw it coming', 'took me by surprise',
        'so surprised', 'i was surprised', 'i am surprised',
        
        # --- EKLENEN KÖK KELİMELER ---
        'shocked', 'stunned', 'astonished', 'amazed', 'astounded',
        'unexpected', 'unbelievable', 'wow'
    ],
    'neutral': [
        'just went', 'just stayed', 'just another day', 'same as always',
        'normal day', 'routine day', 'nothing special', 'as usual',
        'regular day', 'ordinary day', 'did not do much', 'pretty uneventful',
        'nothing happened', 'typical day', 'same old',
        
        # --- EKLENEN KÖK KELİMELER ---
        'uneventful', 'routine', 'typical', 'ordinary', 'average', 'whatever', 'okay'
    ],

}

MODEL_CACHE = 'tfidf_model.pkl'

def train_model():
    global tfidf_vectorizer, tfidf_classifiers

    if not ML_AVAILABLE:
        print("⚠️  scikit-learn not available — keyword fallback only.")
        return

    if os.path.exists(MODEL_CACHE):
        try:
            with open(MODEL_CACHE, 'rb') as f:
                cache = pickle.load(f)
            tfidf_vectorizer  = cache['vectorizer']
            tfidf_classifiers = cache['classifiers']
            print("✅ TF-IDF model loaded from cache.")
            return
        except Exception as e:
            print(f"⚠️  Cache load failed ({e}), re-training…")

    csv_candidates = [
        'emotion_multilabel_balanced_dataset.csv',
        'emotion_multilabel_balanced.csv'
    ]
    csv_path = None
    for name in csv_candidates:
        if os.path.exists(name):
            csv_path = name
            break
    if csv_path is None:
        print("⚠️  No emotion CSV found — keyword fallback only.")
        return

    print(f"📂  Training on: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='text').reset_index(drop=True)

    missing = [c for c in EMOTION_LABELS if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")
        return

    X = df['text'].astype(str).values
    y = df[EMOTION_LABELS].values.astype(int)

    print(f"📊  {len(df)} samples, {len(EMOTION_LABELS)} labels")

    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
        min_df=2,
    )
    X_vec = tfidf_vectorizer.fit_transform(X)

    print("🧠  Training per-label classifiers…")
    for i, label in enumerate(EMOTION_LABELS):
        clf = LogisticRegression(
            C=BEST_C.get(label, 1.0),
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
        )
        clf.fit(X_vec, y[:, i])
        tfidf_classifiers[label] = clf
        print(f"   ✓ {label}")

    try:
        with open(MODEL_CACHE, 'wb') as f:
            pickle.dump({'vectorizer': tfidf_vectorizer, 'classifiers': tfidf_classifiers}, f)
        print(f"💾  Model cached to {MODEL_CACHE}")
    except Exception as e:
        print(f"⚠️  Cache save failed: {e}")

    print("✅  TF-IDF model trained.")

# EMOTION PREDICTION — HYBRID (TF-IDF + Keyword)


def _get_ml_scores(text: str) -> dict:
    """TF-IDF + LR'dan pozitif sınıf olasılığı al."""
    if tfidf_vectorizer is None or not tfidf_classifiers:
        return {l: 1.0 / len(EMOTION_LABELS) for l in EMOTION_LABELS}

    vec_text = tfidf_vectorizer.transform([text])
    scores = {}
    for label in EMOTION_LABELS:
        clf = tfidf_classifiers[label]
        p = clf.predict_proba(vec_text)[0]
        classes = list(clf.classes_)
        pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
        scores[label] = float(p[pos_idx])
    return scores

# 1. Zıtlık belirten İngilizce bağlaçların listesi
CONTRAST_CONJUNCTIONS = [
    r'\buntil\b', r'\bbut\b', r'\bhowever\b', 
    r'\balthough\b', r'\beven though\b', r'\byet\b'
]

def _get_combined_raw_scores(text_chunk: str) -> dict:
    """Verilen metin parçası için Keyword ve ML skorlarını hesaplayıp harmanlar."""
    if not text_chunk.strip():
        return {l: 0.0 for l in EMOTION_LABELS}
        
    text_lower = re.sub(r"[^a-z0-9' ]", " ", text_chunk.lower())
    kw_scores = _get_keyword_scores(text_lower)
    ml_scores = _get_ml_scores(text_chunk)
    
    kw_total = sum(kw_scores.values())
    if kw_total > 0:
        kw_norm = {l: kw_scores[l] / kw_total for l in EMOTION_LABELS}
        return {l: 0.60 * kw_norm[l] + 0.40 * ml_scores[l] for l in EMOTION_LABELS}
    return ml_scores

NEGATION_WORDS = {'not', 'never', 'no', "didn't", "don't", "doesn't", "wasn't", "isn't", "aren't"}

def _get_keyword_scores(text_chunk_lower: str) -> dict:
    """Verilen küçük bir metin parçası (cümle) içindeki kelimeleri olumsuzluk ekiyle kontrol eder."""
    scores = {l: 0.0 for l in EMOTION_LABELS}
    words = text_chunk_lower.split()
    
    for label, kws in STRONG_KEYWORDS.items():
        for kw in kws:
            if kw in text_chunk_lower:
                kw_first_word = kw.split()[0]
                try:
                    idx = words.index(kw_first_word)
                    is_negated = False
                    # Kelimenin 1 veya 2 kelime öncesinde olumsuzluk var mı? (örn: "not very happy")
                    if idx > 0 and words[idx-1] in NEGATION_WORDS:
                        is_negated = True
                    elif idx > 1 and words[idx-2] in NEGATION_WORDS:
                        is_negated = True
                        
                    if is_negated:
                        scores[label] -= 0.5  # Olumsuzsa puanını kır
                    else:
                        scores[label] += 1.0  # Normal eşleşme
                except ValueError:
                    scores[label] += 1.0
                    
    return {k: max(0.0, v) for k, v in scores.items()} # Eksiye düşmesini engelle

def predict_emotions(text: str) -> list:
    """
    Zaman Serisi (Progressive) Duygu Analizi:
    Metni parçalara böler ve sona doğru ilerledikçe duyguların ağırlığını katlayarak artırır.
    """
    try:
        clean = contractions.fix(text)
    except Exception:
        clean = text

    # 1. Metni noktalama işaretlerinden (.!?) veya geçiş bağlaçlarından (but, until) PARÇALARA BÖL
    chunks = re.split(r'(?<=[.!?])\s+|\b(?:but|until|however|although)\b', clean, flags=re.IGNORECASE)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 2] # Boş parçaları temizle
    
    if not chunks:
        chunks = [clean]
        
    total_scores = {l: 0.0 for l in EMOTION_LABELS}
    num_chunks = len(chunks)
    
    # 2. Her parçayı sırasıyla analiz et ve KONUMSAL AĞIRLIK uygula
    for i, chunk in enumerate(chunks):
        pos_ratio = (i + 1) / num_chunks  # Parçanın konumu (0.0 ile 1.0 arası)
        
        # --- SİHRİN GERÇEKLEŞTİĞİ YER (RECENCY BIAS) ---
        if pos_ratio == 1.0:
            weight = 5.0   # EN SON PARÇA: Dev çarpan (Nihai duygu her şeyi domine eder)
        elif pos_ratio > 0.6:
            weight = 1.5   # GELİŞME KISMI: Normalden biraz daha etkili
        else:
            weight = 0.2   # KURULUM KISMI: Çok zayıflatılır (Baştaki "Joy" puanlarını ezer)
            
        chunk_lower = re.sub(r"[^a-z0-9' ]", " ", chunk.lower())
        kw_scores = _get_keyword_scores(chunk_lower)
        ml_scores = _get_ml_scores(chunk)
        
        kw_total = sum(kw_scores.values())
        if kw_total > 0:
            kw_norm = {l: kw_scores[l] / kw_total for l in EMOTION_LABELS}
            chunk_final = {l: 0.60 * kw_norm[l] + 0.40 * ml_scores.get(l, 0) for l in EMOTION_LABELS}
        else:
            chunk_final = ml_scores
            
        # Parçanın skorunu zaman ağırlığıyla çarparak ana toplama ekle
        for l in EMOTION_LABELS:
            total_scores[l] += chunk_final.get(l, 0) * weight

    # 3. Sonuçları normalize et ve top 3'ü döndür
    sorted_probs = sorted(total_scores.items(), key=lambda x: -x[1])
    top3  = sorted_probs[:3]
    total = sum(s for _, s in top3) or 1.0
    
    return [(e, round(s / total, 3)) for e, s in top3]
# KEYWORD FALLBACK (ML yoksa)

EMOTION_KEYWORDS = {
    "joy": [
        ("feel calm", 2.5), ("feel at peace", 2.5), ("feel peaceful", 2.5),
        ("feel good", 2.5), ("feel great", 2.5), ("feel wonderful", 2.5),
        ("feel amazing", 2.5), ("feel happy", 2.5), ("feel better", 2.0),
        ("feeling good", 2.5), ("feeling great", 2.5), ("feeling happy", 2.5),
        ("feeling better", 2.0), ("feeling wonderful", 2.5), ("feeling amazing", 2.5),
        ("calm down", 2.0), ("calmed down", 2.5), ("wind down", 2.0),
        ("doing well", 2.0), ("doing great", 2.5), ("going well", 2.0),
        ("made me smile", 2.5), ("cheered me up", 2.5), ("at peace", 2.5),
        ("happy", 2.5), ("joyful", 2.5), ("cheerful", 2.5), ("delighted", 2.5),
        ("elated", 2.5), ("ecstatic", 3.0), ("excited", 2.5), ("thrilled", 2.5),
        ("grateful", 2.5), ("content", 1.5), ("smile", 1.5), ("laugh", 1.5),
        ("celebrate", 2.5), ("blessed", 2.5), ("overjoyed", 3.0),
        ("beautiful", 2.0), ("wonderful", 2.0), ("amazing", 2.0), ("awesome", 2.0),
        ("peaceful", 2.5), ("calm", 2.0), ("relaxed", 2.0), ("relieved", 2.0),
        ("good", 1.5), ("great", 1.5), ("nice", 1.5), ("fine", 1.2),
    ],
    "sadness": [
        ("feel down", 2.5), ("feeling down", 2.5), ("feel low", 2.5),
        ("feel bad", 3.0), ("feel terrible", 3.0), ("feel awful", 3.0),
        ("feel sad", 2.5), ("feeling sad", 2.5), ("feel empty", 2.5),
        ("feel broken", 2.5), ("not feeling well", 2.5), ("not feeling good", 2.5),
        ("made me sad", 2.5), ("made me cry", 3.0),
        ("sad", 2.5), ("unhappy", 2.5), ("depressed", 3.0), ("miserable", 3.0),
        ("sorrow", 2.5), ("grief", 3.0), ("heartbroken", 3.0),
        ("cry", 2.5), ("crying", 2.5), ("tears", 2.5),
        ("hopeless", 3.0), ("empty", 2.5), ("lonely", 2.5), ("isolated", 2.5),
        ("hurt", 2.5), ("devastated", 3.0), ("disappointed", 2.5),
        ("exhausted", 2.0), ("worthless", 3.0), ("drained", 2.0),
        ("bad", 2.5), ("terrible", 3.0), ("awful", 3.0), ("horrible", 3.0),
        ("down", 1.5),
    ],
    "anger": [
        ("so angry", 3.0), ("makes me angry", 2.5), ("feel angry", 2.5),
        ("fed up", 2.5), ("sick of", 2.0), ("drives me crazy", 2.5),
        ("angry", 2.5), ("anger", 2.5), ("mad", 2.5), ("furious", 3.0),
        ("rage", 3.0), ("hate", 2.5), ("annoyed", 2.5), ("irritated", 2.5),
        ("frustrated", 2.5), ("bitter", 2.0), ("resentful", 2.5), ("unfair", 2.0),
        ("betrayed", 3.0), ("pissed", 3.0),
    ],
    "fear": [
        ("feel anxious", 2.5), ("feeling anxious", 2.5), ("feel scared", 2.5),
        ("feel nervous", 2.5), ("feel stressed", 2.5), ("feeling stressed", 2.5),
        ("on edge", 2.5), ("freaking out", 3.0),
        ("afraid", 2.5), ("scared", 2.5), ("terrified", 3.0),
        ("anxious", 2.5), ("anxiety", 2.5), ("nervous", 2.5), ("worried", 2.5),
        ("dread", 2.5), ("panic", 3.0), ("stressed", 2.5),
        ("insecure", 2.5), ("nightmare", 2.5), ("spiraling", 2.5),
        ("overthinking", 2.5), ("frightened", 2.5), ("petrified", 3.0),
    ],
    "love": [
        ("in love", 3.0), ("head over heels", 3.0), ("love you", 3.0),
        ("love", 2.5), ("adore", 2.5), ("cherish", 2.5), ("affection", 2.0),
        ("devoted", 2.5), ("miss you", 2.5),
    ],
    "surprise": [
        ("cannot believe", 2.5), ("mind blown", 3.0), ("caught off guard", 2.5),
        ("surprised", 2.5), ("shocked", 2.5), ("astonished", 3.0),
        ("amazed", 2.5), ("astounded", 3.0), ("stunned", 2.5),
        ("wow", 2.5), ("unexpected", 2.5), ("unbelievable", 2.0),
    ],
    "neutral": [
        ("nothing special", 1.5), ("just another", 1.2), ("as usual", 1.2),
        ("average day", 1.2), ("same as always", 1.2), ("uneventful", 1.5),
        ("routine", 1.2), ("ordinary", 1.2), ("typical", 1.2),
    ],
}

NEGATIONS = {
    "not", "no", "never", "neither", "nor", "nothing", "nobody",
    "nowhere", "without", "hardly", "barely", "scarcely",
    "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't", "can't", "cannot", "couldn't", "won't",
}
BOOSTERS = {
    "very": 1.8, "so": 1.6, "really": 1.7, "extremely": 2.0,
    "absolutely": 2.0, "totally": 1.7, "completely": 1.8,
    "deeply": 1.8, "incredibly": 1.9, "super": 1.6,
}
CALM_DOWN_NEUTRALIZERS = {
    "calm", "calmed", "calming", "slow", "slowed", "cool", "cooled",
    "wind", "settle", "settled", "ease", "quiet", "quieted",
}

def keyword_fallback(text: str) -> list:
    tokens   = re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()
    raw      = {e: 0.0 for e in EMOTION_KEYWORDS}
    consumed = set()

    def negated(i):
        return any(t in NEGATIONS for t in tokens[max(0, i-3):i])

    def boost(i):
        w = tokens[max(0, i-3):i]
        m = 1.0
        for b, f in BOOSTERS.items():
            if b in w: m = max(m, f)
        return m

    # Pass 1: phrases
    for emotion, kw_list in EMOTION_KEYWORDS.items():
        multi = sorted([(k, w) for k, w in kw_list if len(k.split()) > 1], key=lambda x: -len(x[0]))
        for keyword, base_w in multi:
            kt = keyword.split()
            for start in range(len(tokens) - len(kt) + 1):
                if tokens[start:start+len(kt)] == kt:
                    positions = set(range(start, start+len(kt)))
                    if positions & consumed: continue
                    w = base_w * boost(start)
                    raw[emotion] += -w * 0.5 if negated(start) else w
                    consumed.update(positions)

    # Pass 2: single words
    for emotion, kw_list in EMOTION_KEYWORDS.items():
        singles = [(k, w) for k, w in kw_list if len(k.split()) == 1]
        for keyword, base_w in singles:
            for i, tok in enumerate(tokens):
                if i in consumed: continue
                if tok == keyword:
                    if keyword == "down" and emotion == "sadness":
                        prev = tokens[max(0, i-2):i]
                        if any(pw in CALM_DOWN_NEUTRALIZERS for pw in prev):
                            continue
                    w = base_w * boost(i)
                    raw[emotion] += -w * 0.5 if negated(i) else w

    raw    = {k: max(0.0, v) for k, v in raw.items()}
    ranked = sorted(raw.items(), key=lambda x: -x[1])
    max_s  = ranked[0][1] if ranked else 0

    if max_s > 0:
        above = [(e, s) for e, s in ranked if s >= max_s * 0.25]
    else:
        above = []

    if len(above) >= 3:
        result = above[:3]
    elif len(above) == 2:
        used      = {e for e, _ in above}
        pad       = max(0.01, above[-1][1] * 0.06)
        extras    = [(e, pad) for e, s in ranked if e not in used and s > 0]
        if not extras: extras = [("neutral", pad)]
        result    = above + extras[:1]
    elif len(above) == 1:
        used      = {above[0][0]}
        pad       = max(0.01, above[0][1] * 0.06)
        extras    = [(e, pad) for e, s in ranked if e not in used and s > 0]
        if not extras: extras = [("neutral", pad), ("sadness", pad * 0.5)]
        result    = above + extras[:2]
    else:
        result = [("neutral", 1.0), ("sadness", 0.12), ("joy", 0.08)]

    total = sum(s for _, s in result) or 1.0
    return [(e, round(s / total, 3)) for e, s in result]

EMOTION_VALENCE = {
    "joy":      1.0,
    "love":     0.9,
    "surprise": 0.2,
    "neutral":  0.0,
    "sadness": -0.7,
    "fear":    -0.6,
    "anger":   -0.8,
}
# --- MAİL VE HATIRLATICI SİSTEMİ ---
# GÜVENLİK NOTU: Gerçek bir projede bu şifreler kodun içine yazılmaz, ortam değişkeninden (.env) çekilir!
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
# Artık şifreleri koddan değil, güvenli .env dosyasından çekiyoruz
SENDER_EMAIL = os.getenv("SENDER_EMAIL") 
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

# Kullanıcıların alarm verilerini RAM'de tuttuğumuz geçici sözlük
# Örn: {'Sena': {'email': 'sena@mail.com', 'time': '21:00', 'last_sent': '2026-04-23'}}
active_reminders = {}
base_url = os.getenv("BASE_URL", "http://localhost:5000")

def send_email_reminder(to_email, username):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Take a moment for yourself today 🌿"
    # SENDER_EMAIL'i güvenli .env kasasından çekiyoruz
    sender_email = os.getenv("SENDER_EMAIL")
    msg["From"] = f"Mindiary <{sender_email}>"
    msg["To"] = to_email

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Your Mindiary Reminder</title>
      <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Cinzel:wght@600&family=Raleway:wght@400;600&display=swap" rel="stylesheet">
    </head>
    <body style="margin: 0; padding: 0; background-color: #f4e8d0; font-family: 'Cormorant Garamond', Georgia, serif; color: #2a1408;">

      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f4e8d0; padding: 40px 20px;">
        <tr>
          <td align="center">
            
            <table width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width: 500px; background-color: #fcfaf5; border: 1px solid rgba(200, 160, 80, 0.3); border-radius: 4px; padding: 50px 40px; text-align: center;">
              
              <tr>
                <td align="center" style="padding-bottom: 30px;">
                  <!-- CSS ile değil, doğrudan MINDIARY yazıldı (İ sorunu çözüldü) -->
                  <h1 style="font-family: 'Cinzel', 'Times New Roman', serif; font-size: 22px; letter-spacing: 6px; color: #2a1408; margin: 0; font-weight: 600;">
                    MINDIARY
                  </h1>
                  <div style="width: 40px; height: 1px; background-color: #c8a050; margin: 15px auto 0 auto; opacity: 0.5;"></div>
                </td>
              </tr>

              <tr>
                <td align="left" style="font-size: 18px; line-height: 1.6; color: #2a1408;">
                  <p style="margin: 0 0 20px 0; font-weight: 600;">Hi {username},</p>
                  
                  <!-- Yeni, daha sakin cümlemiz eklendi -->
                  <p style="margin: 0 0 20px 0;">It's time to pause and check in with yourself. Your blank page is ready.</p>
                  
                  <p style="margin: 0 0 30px 0;">Whether today was filled with joy, frustration, or just quiet moments, putting your thoughts into words can bring incredible clarity. Mindiary is here to listen and help you understand your emotional landscape.</p>
                </td>
              </tr>

              <tr>
                <td align="center" style="padding-bottom: 40px;">
                  <a href="{base_url}/frontend.html"style="background-color: #2a1408; color: #f4e8d0; padding: 16px 36px; text-decoration: none; font-family: 'Raleway', Arial, sans-serif; font-size: 11px; letter-spacing: 4px; text-transform: uppercase; font-weight: 600; border-radius: 2px; display: inline-block;">
                    Open My Diary
                  </a>
                </td>
              </tr>

              <tr>
                <td align="center" style="font-size: 18px; font-style: italic; color: rgba(42, 20, 8, 0.8);">
                  Take a deep breath,<br>
                  <span style="font-weight: 600; font-style: normal;">The Mindiary</span>
                </td>
              </tr>

            </table>

            <table width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width: 500px; text-align: center; margin-top: 20px;">
              <tr>
                <td align="center" style="font-family: 'Raleway', Arial, sans-serif; font-size: 10px; line-height: 1.5; color: rgba(42, 20, 8, 0.5); letter-spacing: 1px;">
                  You're receiving this because you set a daily reminder in Mindiary.<br>
                  <a href="{base_url}" style="color: rgba(42, 20, 8, 0.7); text-decoration: underline;">Click here to turn off these reminders.</a>
                </td>
              </tr>
            </table>

          </td>
        </tr>
      </table>

    </body>
    </html>
    """
    
    msg.attach(MIMEText(html_template, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            # Şifreyi de .env kasasından çekiyoruz
            server.login(sender_email, os.getenv("SENDER_PASSWORD"))
            server.sendmail(sender_email, to_email, msg.as_string())
        print(f"📧 BAŞARILI: {to_email} adresine hatırlatıcı maili uçuruldu!")
    except Exception as e:
        print(f"⚠️ MAİL HATASI: {e}")

def reminder_worker():
    """Arka planda her dakika uyanıp saati gelen mailleri gönderen işçi"""
    while True:
        now = datetime.now()
        current_time_str = now.strftime("%H:%M")
        current_date_str = now.strftime("%Y-%m-%d")

        for username, data in list(active_reminders.items()):
            if data['time'] == current_time_str and data.get('last_sent') != current_date_str:
                send_email_reminder(data['email'], username)
                data['last_sent'] = current_date_str

        time.sleep(60) # Sistemi yormamak için 60 saniyede bir kontrol et

# Sunucu başlarken arka plan işçisini de başlatıyoruz
threading.Thread(target=reminder_worker, daemon=True).start()

def compute_wellness_score(emotions: list) -> float:
    if not emotions: return 5.0
    total_w = sum(s for _, s in emotions)
    if total_w == 0: return 5.0
    wv = sum(EMOTION_VALENCE.get(e, 0) * s for e, s in emotions) / total_w
    return round(max(1.0, min(10.0, (wv + 1) / 2 * 9 + 1)), 1)

def _intensity_tier(score: float) -> str:
    if score >= 0.65:   return "intense"
    elif score >= 0.35: return "moderate"
    return "mild"

CAT_JSON_KEY = {
    "food":                 "food",
    "activity":             "activities",
    "spotify_playlists":    "spotify_playlists",
    "places":               "places",
    "colors":               "colors",
    "shows_movies":         "shows_movies",
    "social":               "social",
    "personal_development": "personal_development",
}
ALL_CATEGORIES = list(CAT_JSON_KEY.keys())

def get_dynamic_movies(emotion: str) -> list:
    """TMDB API kullanarak anlık duyguya özel hem film hem DİZİ önerilerini KARIŞIK çeker."""
    
    
    genre_map = {
        'joy': '35',            # Komedi
        'sadness': '18',        # Drama
        'anger': '28,10759',    # Aksiyon (Film 28, Dizi 10759)
        'fear': '16,10751',     # Animasyon, Aile
        'surprise': '878,10765',# Bilim Kurgu (Film 878, Dizi 10765)
        'love': '10749,18',     # Romantik
        'neutral': '99'         # Belgesel
    }
    
    # API anahtarını gizli .env dosyasından çekiyoruz
    api_key = os.getenv("TMDB_API_KEY")
    
    if not api_key or api_key == "SENIN_TMDB_API_ANAHTARIN":
        return []

    genres = genre_map.get(emotion, '35')
    
    # 1. Hem FİLM hem DİZİ (TV) için ayrı iki istek hazırlıyoruz
    movie_url = f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}&with_genres={genres}&vote_average.gte=7.0&sort_by=popularity.desc"
    tv_url = f"https://api.themoviedb.org/3/discover/tv?api_key={api_key}&with_genres={genres}&vote_average.gte=7.0&sort_by=popularity.desc"
    
    try:
        # 2. İki API'ye de aynı anda istek atıyoruz
        movies = requests.get(movie_url, timeout=3).json().get('results', [])
        shows = requests.get(tv_url, timeout=3).json().get('results', [])
        
        # 3. Hangi verinin ne olduğunu Frontend anlasın diye etiketliyoruz
        for m in movies: m['media_type'] = 'film'
        for s in shows: s['media_type'] = 'series'
        
        # 4. Tüm dizi ve filmleri TEK BİR HAVUZDA birleştiriyoruz
        combined_pool = movies + shows
        
        # 5. Bu büyük havuzdan rastgele 4 tane seçiyoruz (Böylece hep karışık gelecek)
        selected = random.sample(combined_pool, min(4, len(combined_pool)))
        
        results = []
        for item in selected:
            title = item.get('title') or item.get('name') or item.get('original_title')
            imdb = str(round(item.get('vote_average', 0.0), 1)) + "/10"
            m_type = item['media_type']
            
            # Google linki için uygun arama terimi ekliyoruz
            search_suffix = "tv show" if m_type == "series" else "movie"
            search_query = urllib.parse.quote(f"{title} {search_suffix}")
            google_url = f"https://www.google.com/search?q={search_query}"
            
            results.append({
                "type": m_type, 
                "title": title,
                "imdb": imdb,
                "url": google_url,
                "why": f"This highly-rated {search_suffix} was dynamically selected to match your emotional state."
            })
        
        return results
    except Exception as e:
        print(f"⚠️ KRİTİK HATA: İstek atılırken çöktü! Detay: {e}")
        return []
    
def get_dynamic_food(emotion: str) -> list:
    
    
    ingredient_map = {
        'joy': ['mango', 'strawberry', 'citrus', 'blueberry', 'peach'],
        'sadness': ['dark chocolate', 'banana', 'oatmeal', 'soup', 'cinnamon'],
        'anger': ['crunchy almond', 'green tea', 'carrot', 'apple', 'mint'],
        'fear': ['chamomile', 'spinach', 'avocado', 'salmon', 'warm milk'],
        'love': ['chocolate', 'honey', 'strawberry', 'cherry', 'vanilla'],
        'surprise': ['spicy', 'curry', 'jalapeno', 'ginger', 'lime'],
        'neutral': ['avocado', 'lentil', 'chicken', 'toast', 'egg']
    }
    
    # API anahtarını gizli .env dosyasından çekiyoruz
    api_key = os.getenv("SPOONACULAR_API_KEY")
    
    if not api_key or api_key == "SENIN_SPOONACULAR_API_ANAHTARIN":
        return []

    pool = ingredient_map.get(emotion, ['apple', 'banana', 'honey'])
    selected_ingredients = random.sample(pool, min(3, len(pool)))
    
    results = []
    print(f"\n" + "="*40)
    print(f"🍳 SPOONACULAR ZAMAN-BAZLI ARAMA TETİKLENDİ")
    
    for query in selected_ingredients:
        # addRecipeInformation=true ile tarif sürelerini (readyInMinutes) de çekiyoruz!
        url = f"https://api.spoonacular.com/recipes/complexSearch?query={query}&number=8&addRecipeInformation=true&apiKey={api_key}"
        
        try:
            resp = requests.get(url, timeout=4)
            if resp.status_code == 200:
                data = resp.json().get('results', [])
                
                # 30 dk ve altı olanları HIZLI, 30 dk üstü olanları YAVAŞ/TERAPİ olarak ayırıyoruz
                easy_cands = [r for r in data if r.get('readyInMinutes', 999) <= 30]
                hard_cands = [r for r in data if r.get('readyInMinutes', 0) > 30]
                
                if easy_cands:
                    r = random.choice(easy_cands)
                    title = r.get('title')
                    time_mins = r.get('readyInMinutes', 15)
                    google_url = f"https://www.google.com/search?q={urllib.parse.quote(title + ' recipe')}"
                    results.append({
                        "item": title,
                        "url": google_url,
                        "time": time_mins,
                        "difficulty": "easy",
                        "why": f"Quick & Easy ({time_mins} min): Uses '{query}' to gently balance your mood without demanding too much energy."
                    })
                    print(f"   ⚡ PRATİK ({time_mins}dk): {title}")
                    
                if hard_cands:
                    r = random.choice(hard_cands)
                    title = r.get('title')
                    time_mins = r.get('readyInMinutes', 60)
                    google_url = f"https://www.google.com/search?q={urllib.parse.quote(title + ' recipe')}"
                    results.append({
                        "item": title,
                        "url": google_url,
                        "time": time_mins,
                        "difficulty": "hard",
                        "why": f"Cooking Therapy ({time_mins} min): An immersive recipe with '{query}'. The repetitive tasks help ground your mind."
                    })
                    print(f"   🧘 TERAPÖTİK ({time_mins}dk): {title}")
        except Exception as e:
            print(f"⚠️ Yemek API Hata ({query}): {e}")
            
    print("="*40 + "\n")
    return results

def get_full_recommendations(text: str, emotions: list, location: str = "", rag_boosted: set = None):
    if rag_boosted is None:
        rag_boosted = set()
        
    normalized = []
    for e in emotions:
        if isinstance(e, dict):
            normalized.append((e["name"], float(e.get("score", 0.33))))
        else:
            normalized.append((str(e[0]), float(e[1])))
    if not normalized:
        normalized = [("neutral", 1.0)]

    dominant_emotion, dominant_score = normalized[0]
    tier = _intensity_tier(dominant_score)
    db   = WELLNESS_DATA.get("emotions", {})

    note = ""

    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
    rng  = random.Random(seed)

    def rag_score(item):
        """Return a small boost score if item was previously rated highly."""
        label = ""
        if isinstance(item, dict):
            label = (item.get("item") or item.get("name") or item.get("title") or "").lower().strip()
        elif isinstance(item, str):
            label = item.lower().strip()
        
        return 0.5 if label in rag_boosted else 0.0

    def pick_from(emo_name, api_category, n=3):
        json_key   = CAT_JSON_KEY.get(api_category, api_category)
        emo_block  = db.get(emo_name) or db.get("neutral") or {}
        tiers_order = [tier] + [t for t in ["mild", "moderate", "intense"] if t != tier]
        for t in tiers_order:
            items = emo_block.get(t, {}).get(json_key, [])
            if items:
                shuffled = list(items)
                rng.shuffle(shuffled)
                # Boost RAG-favoured items to front
                shuffled.sort(key=lambda x: -rag_score(x))
                return shuffled[:n]
        neutral_block = db.get("neutral", {})
        for t in tiers_order:
            items = neutral_block.get(t, {}).get(json_key, [])
            if items:
                shuffled = list(items)
                rng.shuffle(shuffled)
                shuffled.sort(key=lambda x: -rag_score(x))
                return shuffled[:n]
        return []

    recs = {}

    for cat in ALL_CATEGORIES:
        if cat == "spotify_playlists":
            recs[cat] = pick_from(dominant_emotion, "spotify_playlists", n=2)
            continue

        if cat == "places":
            place_items = pick_from(dominant_emotion, "places", n=1)
            pd_item     = place_items[0] if place_items and isinstance(place_items[0], dict) else {}
            default_q   = ["park", "cafe", "nature walk"]
            queries     = pd_item.get("search_queries", default_q)
            recs[cat]   = {
                "description":    pd_item.get("description", "Peaceful outdoor spaces to restore wellbeing."),
                "types":          pd_item.get("types", ["park", "cafe", "nature walk"]),
                "search_queries": queries,
                "location":       location or "",
                "maps_search":    _build_maps_search(queries, location),
            }
            continue

        # Dizi ve Filmler için Dinamik API Entegrasyonu
        if cat == "shows_movies":
            dynamic_shows = get_dynamic_movies(dominant_emotion)
            # Eğer API'den film geldiyse onu kullan
            if dynamic_shows:
                recs[cat] = dynamic_shows
            else:
                # API başarısız olursa veya key yoksa eski JSON yedeğine (Fallback) dön
                pool = {}
                for emo_name, emo_score in normalized[:2]:
                    for item in pick_from(emo_name, cat, n=4):
                        key = json.dumps(item, sort_keys=True)
                        if key not in pool:
                            pool[key] = {"item": item, "score": emo_score + rag_score(item)}
                        else:
                            pool[key]["score"] += emo_score * 0.5
                sorted_items = sorted(pool.values(), key=lambda x: -x["score"])
                recs[cat] = [v["item"] for v in sorted_items[:4]]
            continue

        if cat in ("colors", "social", "personal_development"):
            pool   = {}
            limits = {"colors": 3, "social": 3, "personal_development": 3}
            
            for emo_name, emo_score in normalized[:2]:
                for item in pick_from(emo_name, cat, n=4):
                    key = json.dumps(item, sort_keys=True)
                    if key not in pool:
                        pool[key] = {"item": item, "score": emo_score + rag_score(item)}
                    else:
                        pool[key]["score"] += emo_score * 0.5
            
            sorted_items = sorted(pool.values(), key=lambda x: -x["score"])
            recs[cat]    = [v["item"] for v in sorted_items[:limits[cat]]]
            continue


        # food & activity — blend top-3 emotions
        # YEMEKLER İÇİN DİNAMİK API
        if cat == "food":
            dynamic_food = get_dynamic_food(dominant_emotion)
            if dynamic_food:
                recs[cat] = dynamic_food
            else:
                # API başarısız olursa JSON'a (Fallback) dön
                pool = {}
                for emo_name, emo_score in normalized[:3]:
                    for item in pick_from(emo_name, cat, n=3):
                        key = json.dumps(item, sort_keys=True)
                        if key not in pool: pool[key] = {"item": item, "score": emo_score + rag_score(item)}
                        else: pool[key]["score"] += emo_score * 0.3
                sorted_items = sorted(pool.values(), key=lambda x: -x["score"])
                recs[cat] = [v["item"] for v in sorted_items[:3]]
            continue

        # AKTİVİTELER İÇİN ESKİ JSON MANTIĞI
        if cat == "activity":
            pool = {}
            for emo_name, emo_score in normalized[:3]:
                for item in pick_from(emo_name, cat, n=4):
                    key = json.dumps(item, sort_keys=True)
                    if key not in pool: pool[key] = {"item": item, "score": emo_score + rag_score(item)}
                    else: pool[key]["score"] += emo_score * 0.3
            sorted_items = sorted(pool.values(), key=lambda x: -x["score"])
            recs[cat] = [v["item"] for v in sorted_items[:3]]
            continue

    if note is None:
        emo_block = db.get(dominant_emotion, {})
        note = emo_block.get(tier, {}).get("science_note")

    for cat in ALL_CATEGORIES:
        if cat not in recs:
            recs[cat] = [] if cat != "places" else {
                "description": "Find a calm outdoor space near you.",
                "types": ["park", "cafe"],
                "search_queries": ["park", "cafe"],
                "location": location,
                "maps_search": _build_maps_search(["park", "cafe"], location),
            }

    return recs, note, tier

def _build_maps_search(search_queries: list, location: str) -> list:
    maps_links = []
    for query in search_queries[:3]:
        if location:
            search_str = f"{query.replace(' ', '+')}+in+{location.replace(' ', '+')}"
        else:
            search_str = query.replace(' ', '+')
        maps_links.append({
            "label": query.title(),
            "url":   f"https://www.google.com/maps/search/{search_str}",
            "query": f"{query} {location}".strip(),
        })
    return maps_links

@app.route('/auth/register', methods=['POST'])
def register():
    data     = request.get_json()
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters."}), 400
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, pw_hash) VALUES (%s, %s)",
                    (username, hash_password(password)))
        conn.commit()
        cur.execute("SELECT id, username, created_at FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        return jsonify({"message": "Account created.", "user": user}), 201
    except psycopg2.IntegrityError:
        return jsonify({"error": "Username already taken."}), 409
    finally:
        conn.close()

@app.route('/auth/login', methods=['POST'])
def login():
    data     = request.get_json()
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT id, username, pw_hash FROM users WHERE username=%s", (username,))
    user = fetchone_dict(cur)
    conn.close()
    if not user or user['pw_hash'] != hash_password(password):
        return jsonify({"error": "Wrong username or password."}), 401
    return jsonify({
        "message": "Login successful.",
        "user": {"id": user["id"], "username": user["username"]}
    }), 200

@app.route('/feedback', methods=['POST'])
def save_feedback():
    data = request.get_json()
    username = data.get('username')
    item_text = data.get('item_text')
    category = data.get('category', 'general')
    did_it = data.get('did_it', True)
    stars = data.get('stars', 0)
    feedback_text = data.get('feedback_text', '')
    emotion = data.get('emotion', 'neutral')

    if not username or not item_text:
        return jsonify({"error": "Missing username or item_text."}), 400

    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        if not user:
            return jsonify({"error": "User not found."}), 404

        cur.execute("""
            INSERT INTO user_feedback 
                (user_id, item_text, category, did_it, stars, feedback_text, emotion)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user['id'], item_text, category, 1 if did_it else 0, stars, feedback_text, emotion))
        conn.commit()
        return jsonify({"message": "Feedback saved successfully.", "status": "ok"}), 200
    finally:
        conn.close()

# ANALYZE + SAVE ENTRY

@app.route('/analyze', methods=['POST'])
def analyze():
    data        = request.get_json()
    username    = data.get('username', '')
    text        = data.get('text', '')
    location    = data.get('location', '')
    page_number = data.get('page_number', 1)
    entry_date  = data.get('entry_date', datetime.now().date().isoformat())

    if not text:
        return jsonify({"error": "No text provided."}), 400

    rag_boosted = set()
    user_id = None
    
    if username:
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM users WHERE username=%s", (username,))
            user = fetchone_dict(cur)
            if user:
                user_id = user['id']
                cur.execute("""
                    SELECT item_text 
                    FROM user_feedback 
                    WHERE user_id=%s AND did_it=1
                    GROUP BY item_text
                    HAVING AVG(CAST(stars AS FLOAT)) >= 3.0
                """, (user_id,))
                
                for row in fetchall_dict(cur):
                    rag_boosted.add(row['item_text'].lower().strip())
        finally:
            conn.close()

    if tfidf_vectorizer and tfidf_classifiers:
        emotions_list = predict_emotions(text)
    else:
        emotions_list = keyword_fallback(text)

    recs, note, tier = get_full_recommendations(text, emotions_list, location, rag_boosted=rag_boosted)
    wellness_score   = compute_wellness_score(emotions_list)

    entry_id = None
    if username and user_id:
        conn = get_db()
        cur  = conn.cursor()
        try:
            cur.execute(
                "SELECT id FROM entries WHERE user_id=%s AND page_number=%s",
                (user_id, page_number))
            existing = fetchone_dict(cur)
            now = datetime.now().isoformat()

            if existing:
                entry_id = existing['id']
                cur.execute("""
                    UPDATE entries
                    SET entry_text=%s, location=%s, entry_date=%s,
                        wellness_note=%s, mood_score=%s, updated_at=%s
                    WHERE id=%s
                """, (text, location, entry_date, note, wellness_score, now, entry_id))
                cur.execute("DELETE FROM emotions WHERE entry_id=%s", (entry_id,))
                cur.execute("DELETE FROM recommendations WHERE entry_id=%s", (entry_id,))
            else:
                cur.execute("""
                    INSERT INTO entries
                        (user_id, page_number, entry_date, location,
                        entry_text, wellness_note, mood_score, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (user_id, page_number, entry_date, location, text, note, wellness_score, now, now))
                entry_id = cur.fetchone()['id']

            for emo_name, score in emotions_list:
                cur.execute(
                    "INSERT INTO emotions (entry_id, emotion, score) VALUES (%s, %s, %s)",
                    (entry_id, emo_name, score))

            for category, items in recs.items():
                item_json = json.dumps(items, ensure_ascii=False)
                cur.execute(
                    "INSERT INTO recommendations (entry_id, category, item, why) "
                    "VALUES (%s, %s, %s, %s)",
                    (entry_id, category, item_json, ""))

            conn.commit()
        finally:
            conn.close()

    return jsonify({
        "entry_id":        entry_id,
        "emotions":        [{"name": n, "score": s} for n, s in emotions_list],
        "recommendations": recs,
        "wellness_note":   note,
        "wellness_score":  wellness_score,
        "tier":            tier,
        "status":          "ok",
        "model":           "tfidf_hybrid" if tfidf_vectorizer else "keyword_fallback",
        "rag_active":      len(rag_boosted) > 0
    })

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(f"Unhandled error: {e}")
    traceback.print_exc()
    return jsonify({
        "error":   str(e),
        "status":  "error",
        "message": "Server encountered an error."
    }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "status": "error"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "status": "error"}), 405

@app.route('/places/suggest', methods=['POST'])
def suggest_places():
    data     = request.get_json()
    emotion  = data.get('emotion', 'neutral')
    tier     = data.get('tier', 'mild')
    location = data.get('location', '')
    db       = WELLNESS_DATA.get("emotions", {})
    emo_block  = db.get(emotion, db.get("neutral", {}))
    tier_block = emo_block.get(tier, emo_block.get("mild", {}))
    place_data = tier_block.get("places", {})
    if not place_data:
        return jsonify({
            "description": "Peaceful outdoor spaces to restore wellbeing",
            "types":       ["park", "cafe", "nature walk"],
            "maps_links":  _build_maps_search(["park", "cafe"], location),
            "location":    location,
        })
    return jsonify({
        "description": place_data.get("description", ""),
        "types":       place_data.get("types", []),
        "maps_links":  _build_maps_search(place_data.get("search_queries", []), location),
        "location":    location,
        "emotion":     emotion,
        "tier":        tier,
    })

@app.route('/pages/<username>', methods=['GET'])
def get_pages(username):
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        if not user:
            return jsonify({"error": "User not found."}), 404

        cur.execute("""
            SELECT id, page_number, entry_date, location,
                   entry_text, wellness_note, mood_score, created_at, updated_at, card_feedback
            FROM entries WHERE user_id=%s ORDER BY page_number ASC
        """, (user['id'],))
        entries = fetchall_dict(cur)

        pages = []
        for entry in entries:
            eid = entry['id']
            cur.execute(
                "SELECT emotion, score FROM emotions WHERE entry_id=%s ORDER BY score DESC", (eid,))
            emotions_rows = fetchall_dict(cur)
            cur.execute(
                "SELECT category, item, why FROM recommendations WHERE entry_id=%s", (eid,))
            rec_rows = fetchall_dict(cur)

            recs = {}
            # EKLENEN KISIM: Veritabanındaki önerileri (recommendations) çözüp frontend'e yolluyoruz.
            for r in rec_rows:
                try:
                    recs[r['category']] = json.loads(r['item'])
                except Exception:
                    recs[r['category']] = []

            try:
                cf = json.loads(entry['card_feedback']) if entry['card_feedback'] else {}
            except Exception:
                cf = {}

            pages.append({
                "id":              eid,
                "page_number":     entry['page_number'],
                "date":            entry['entry_date'],
                "location":        entry['location'] or '',
                "text":            entry['entry_text'],
                "wellness_note":   entry['wellness_note'],
                "wellness_score":  entry['mood_score'],
                "analyzed":        True,
                "emotions":        [{"name": r['emotion'], "score": r['score']} for r in emotions_rows],
                "recommendations": recs,
                "cardFeedback":    cf,  # <--- BU SATIRI EKLEYİN
                "tier": _intensity_tier(entry['mood_score']) if entry.get('mood_score') is not None else "mild",                "created_at":      entry['created_at'],
                "updated_at":      entry['updated_at'],
            })

        return jsonify({"username": username, "pages": pages})
    finally:
        conn.close()

@app.route('/pages/<username>/<int:page_number>', methods=['DELETE'])
def delete_page(username, page_number):
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        if not user:
            return jsonify({"error": "User not found."}), 404
        cur.execute(
            "DELETE FROM entries WHERE user_id=%s AND page_number=%s",
            (user['id'], page_number))
        conn.commit()
        return jsonify({"message": f"Page {page_number} deleted."})
    finally:
        conn.close()

@app.route('/survey', methods=['POST'])
def submit_survey():
    data           = request.get_json()
    username       = data.get('username', '')
    mood_score     = data.get('mood_score')
    liked_items    = data.get('liked_items', [])
    disliked_items = data.get('disliked_items', [])
    free_text      = data.get('free_text', '')
    entry_id       = data.get('entry_id')

    if not username or mood_score is None:
        return jsonify({"error": "username and mood_score are required."}), 400

    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        if not user:
            return jsonify({"error": "User not found."}), 404

        today = datetime.date.today().isoformat()
        cur.execute(
            "SELECT id FROM surveys WHERE user_id=%s AND survey_date=%s",
            (user['id'], today))
        existing = fetchone_dict(cur)

        if existing:
            cur.execute("""
                UPDATE surveys
                SET mood_score=%s, liked_items=%s, disliked_items=%s, free_text=%s, entry_id=%s
                WHERE id=%s
            """, (int(mood_score), json.dumps(liked_items), json.dumps(disliked_items),
                  free_text, entry_id, existing['id']))
        else:
            cur.execute("""
                INSERT INTO surveys
                    (user_id, entry_id, survey_date, mood_score,
                     liked_items, disliked_items, free_text)
                VALUES (%s, %s, %s ,%s, %s, %s,%s)
            """, (user['id'], entry_id, today, int(mood_score),
                  json.dumps(liked_items), json.dumps(disliked_items), free_text))

        conn.commit()
        return jsonify({"message": "Survey saved.", "wellness_score": mood_score})
    finally:
        conn.close()

@app.route('/survey/<username>', methods=['GET'])
def get_surveys(username):
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        if not user:
            return jsonify({"error": "User not found."}), 404

        cur.execute("""
            SELECT survey_date, mood_score, liked_items, disliked_items, free_text, entry_id
            FROM surveys WHERE user_id=%s ORDER BY survey_date ASC
        """, (user['id'],))
        rows = fetchall_dict(cur)
        return jsonify({
            "username": username,
            "surveys": [{
                "date":           r['survey_date'],
                "mood_score":     r['mood_score'],
                "liked_items":    json.loads(r['liked_items'] or '[]'),
                "disliked_items": json.loads(r['disliked_items'] or '[]'),
                "free_text":      r['free_text'],
                "entry_id":       r['entry_id'],
            } for r in rows]
        })
    finally:
        conn.close()

@app.route('/stats/<username>', methods=['GET'])
def get_stats(username):
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        if not user:
            return jsonify({"error": "User not found."}), 404

        cur.execute("""
            SELECT e.emotion, AVG(e.score) as avg_score, COUNT(*) as cnt
            FROM emotions e JOIN entries en ON e.entry_id = en.id
            WHERE en.user_id = %s
            GROUP BY e.emotion ORDER BY avg_score DESC
        """, (user['id'],))
        emotion_rows = fetchall_dict(cur)

        cur.execute("SELECT COUNT(*) AS n FROM entries WHERE user_id=%s", (user['id'],))
        # total_entries = cur.fetchone()[0] YERİNE:
        total_entries = cur.fetchone()['n']

        cur.execute("""
            SELECT entry_date, mood_score FROM entries
            WHERE user_id=%s AND mood_score IS NOT NULL ORDER BY entry_date ASC
        """, (user['id'],))
        trend_rows = fetchall_dict(cur)

        return jsonify({
            "username":       username,
            "total_entries":  total_entries,
            "emotion_stats":  [
                {"emotion": r['emotion'], "avg_score": round(r['avg_score'], 3), "count": r['cnt']}
                for r in emotion_rows
            ],
            "wellness_trend": [
                {"date": r['entry_date'], "score": r['mood_score']}
                for r in trend_rows
            ],
        })
    finally:
        conn.close()

@app.route('/debug/predict', methods=['GET'])
def debug_predict():
    text     = request.args.get('text', 'I feel happy but also a little anxious')
    location = request.args.get('location', 'Istanbul')

    if tfidf_vectorizer and tfidf_classifiers:
        emotions = predict_emotions(text)
        model_used = "tfidf_hybrid"
    else:
        emotions = keyword_fallback(text)
        model_used = "keyword_fallback"

    score = compute_wellness_score(emotions)
    recs, note, tier = get_full_recommendations(text, emotions, location)
    return jsonify({
        "input":            text,
        "model_used":       model_used,
        "tfidf_loaded":     tfidf_vectorizer is not None,
        "wellness_score":   score,
        "wellness_note":    note,
        "tier":             tier,
        "emotions":         [{"name": e, "score": round(s, 4)} for e, s in emotions],
        "categories":       list(recs.keys()),
        "category_counts":  {k: len(v) if isinstance(v, list) else "object" for k, v in recs.items()},
        "recommendations":  recs,
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":       "ok",
        "model":        "tfidf_hybrid" if tfidf_vectorizer else "keyword_fallback",
        "tfidf_loaded": tfidf_vectorizer is not None,
        "labels":       EMOTION_LABELS,
    })

@app.route('/')
def home():
    return render_template('index.html')

# frontend.html sayfasını sunmak için gereken yönlendirme (route)
@app.route('/frontend.html')
def frontend_page():
    return render_template('frontend.html')

@app.route('/journal')
def journal():
    return render_template('frontend.html')

@app.route('/pages/<username>/<int:page_number>/card_feedback', methods=['POST'])
def update_card_feedback(username, page_number):
    data = request.get_json()
    cf_json = json.dumps(data.get('cardFeedback', {}))
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        if user:
            cur.execute("""
                UPDATE entries SET card_feedback=%s 
                WHERE user_id=%s AND page_number=%s
            """, (cf_json, user['id'], page_number))
            conn.commit()
        return jsonify({"status": "ok"})
    finally:
        conn.close()

@app.route('/feedback/<username>', methods=['GET'])
def get_feedback(username):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = fetchone_dict(cur)
        if not user:
            return jsonify({"error": "User not found."}), 404
        cur.execute("""
            SELECT item_text, category, did_it, stars, feedback_text, emotion 
            FROM user_feedback WHERE user_id=%s
        """, (user['id'],))
        return jsonify({"username": username, "feedback": fetchall_dict(cur)})
    finally:
        conn.close()

@app.route('/api/reminder/set', methods=['POST'])
def set_reminder():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    time_str = data.get('time') # "21:00" formatında gelecek

    # Basit bir format kontrolü ve güvenlik
    if not username or not email or not time_str:
        return jsonify({"error": "Missing information"}), 400

    # Saatin sadece "HH:MM" formatında olduğundan emin ol (HTML time input'u saniye de gönderebilir)
    time_parts = time_str.split(":")
    if len(time_parts) >= 2:
        clean_time_str = f"{time_parts[0]}:{time_parts[1]}"
    else:
        clean_time_str = time_str

    # Alarmı RAM'e (veya ileride veritabanına) kaydet
    active_reminders[username] = {'email': email, 'time': clean_time_str}
    
    print(f"⏰ REMINDER SET: Emails will be sent to {email} daily at {clean_time_str} for {username}.")
    return jsonify({"status": "ok", "message": "Reminder set successfully."})

# CORS header on every response

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
    return response

if __name__ == '__main__':
    init_db()
    train_model()
    print("🚀 Server running → http://localhost:5000")
    app.run(debug=False, port=5000)