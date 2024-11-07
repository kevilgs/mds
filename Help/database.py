import mysql.connector

def create_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='1234',
        database='mds'
    )
    return conn

def create_tables():
    conn = create_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            name VARCHAR(255),
            email VARCHAR(255)
        )
    ''')
    
    # Create user_info table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_info (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            age INT,
            overweight VARCHAR(255),
            smoke VARCHAR(255),
            injured VARCHAR(255),
            cholesterol VARCHAR(255),
            hypertension VARCHAR(255),
            diabetes VARCHAR(255),
            symptoms TEXT,
            predicted_disease VARCHAR(255),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_user(username, password, name, email):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, password, name, email) VALUES (%s, %s, %s, %s)', (username, password, name, email))
    conn.commit()
    conn.close()

def get_user(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def add_user_info(user_id, age, overweight, smoke, injured, cholesterol, hypertension, diabetes, symptoms, predicted_disease):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_info (user_id, age, overweight, smoke, injured, cholesterol, hypertension, diabetes, symptoms, predicted_disease)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (user_id, age, overweight, smoke, injured, cholesterol, hypertension, diabetes, symptoms, predicted_disease))
    conn.commit()
    conn.close()

# Initialize the database and create tables
create_tables()