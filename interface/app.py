import json
import os
import time
import uuid
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import streamlit as st
from kafka import KafkaProducer

KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions"),
}

def load_file(uploaded_file_):
    try:
        return pd.read_csv(uploaded_file_)
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None

def send_to_kafka(df, topic, bootstrap_servers):
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        df['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        progress_bar = st.progress(0)
        for idx, row in df.iterrows():
            producer.send(topic, value={
                "transaction_id": row['transaction_id'],
                "data": row.drop('transaction_id').to_dict(),
            })
            progress_bar.progress((idx + 1) / len(df))
            time.sleep(0.01)
        producer.flush()
        return True
    except Exception as e:
        st.error(f"Ошибка отправки: {e}")
        return False

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

def load_fraud_transactions(limit=10):
    conn = get_db_connection()
    query = f"""
        SELECT transaction_id, score, fraud_flag, us_state, merch, cat_id, created_at
        FROM scores WHERE fraud_flag = 1
        ORDER BY created_at DESC LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_scores(limit=100):
    conn = get_db_connection()
    query = f"SELECT score FROM scores ORDER BY created_at DESC LIMIT {limit}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

st.title("Realtime Fraud Detector")

uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
    st.session_state.uploaded_files[uploaded_file.name] = {
        "status": "Загружен",
        "df": load_file(uploaded_file),
    }
    st.success(f"Файл {uploaded_file.name} загружен")

if st.session_state.uploaded_files:
    st.subheader("Список файлов")
    for file_name, file_data in st.session_state.uploaded_files.items():
        cols = st.columns([4, 2, 2])
        with cols[0]:
            st.markdown(f"**Файл:** `{file_name}`")
            st.markdown(f"**Статус:** `{file_data['status']}`")
        with cols[2]:
            if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                if file_data["df"] is not None:
                    with st.spinner("Отправка..."):
                        success = send_to_kafka(
                            file_data["df"],
                            KAFKA_CONFIG["topic"],
                            KAFKA_CONFIG["bootstrap_servers"],
                        )
                        if success:
                            st.session_state.uploaded_files[file_name]["status"] = "Отправлен"
                            st.rerun()
                else:
                    st.error("Файл пуст")

if st.button("Посмотреть результаты"):
    st.subheader("Последние 10 фродовых транзакций")
    fraud_df = load_fraud_transactions(limit=10)
    if not fraud_df.empty:
        st.dataframe(fraud_df)
    else:
        st.write("Нет записей с fraud_flag=1")

    st.subheader("Гистограмма скоров последних 100 транзакций")
    score_df = load_scores(limit=100)
    if not score_df.empty:
        fig, ax = plt.subplots()
        ax.hist(score_df['score'], bins=40, color='steelblue', edgecolor='black')
        ax.set_title('Распределение скоров')
        ax.set_xlabel('Score')
        ax.set_ylabel('Частота')
        ax.grid(ls=':')
        st.pyplot(fig)
    else:
        st.write("Нет записей в базе")