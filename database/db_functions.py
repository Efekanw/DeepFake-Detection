import psycopg2
import json
import hashlib
import binascii
import os


def check_login(connection, username, password):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT password, id FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        print(result)
        if result:
            # parolayı hash'leme ve veritabanındaki hash ile karşılaştırma
            stored_password = bytes(result[0])
            salt = stored_password[:64].decode('ascii')
            stored_password = stored_password[64:].decode('ascii')

            # Şifreyi hashleme ve karşılaştırma
            hashed_password = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt.encode('ascii'), 100000)
            hashed_password = binascii.hexlify(hashed_password).decode('ascii')
            if stored_password == hashed_password:
                return result[1]
            else:
                return False
        else:
            return False
    except(Exception, psycopg2.Error) as errorMsg:
        print("A database-related error occured: ", errorMsg)
        return []


def register(connection, username, hashedpassword, mail):
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users( username, password, mail) VALUES( %s, %s, %s)", (username, hashedpassword.encode('utf-8'), mail, ))
        connection.commit()
        cursor.close()
    except(Exception, psycopg2.Error) as errorMsg:
        print("A database-related error occured: ", errorMsg)
        return []


def insert_metrics(connection, metric_name, conf_image_path, userid, roc_image_path, accuracy, f1, loss, model_path):
    try:
        cursor = connection.cursor()

        with open(conf_image_path, 'rb') as file:
            conf_image_data = file.read()

        with open(roc_image_path, 'rb') as file:
            roc_image_data = file.read()

        # Check if the metric_name already exists

            # Metric doesn't exist, insert a new row
        accuracy = float(accuracy)
        f1 = float(f1)
        loss = float(loss)
        insert_query = "INSERT INTO metrics (conf_image, user_id, roc_image, accuracy, f1, loss, metric_name, model_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"
        cursor.execute(insert_query, (
        psycopg2.Binary(conf_image_data), userid, psycopg2.Binary(roc_image_data), accuracy, f1, loss, metric_name, model_path))

        connection.commit()
        print("Metric inserted or updated successfully in PostgreSQL")
    except (Exception, psycopg2.Error) as errorMsg:
        print("Error while inserting or updating metric in PostgreSQL:", errorMsg)


def retrieve_conf_image(connection, metric_id):
    try:
        cursor = connection.cursor()

        retrieve_query = "SELECT conf_image FROM metrics WHERE id = %s;"
        cursor.execute(retrieve_query, (metric_id,))
        image_data = cursor.fetchone()[0]
        print("Conf image retrieved successfully from PostgreSQL")
        return image_data
    except (Exception, psycopg2.Error) as errorMsg:
        print("Error while retrieving image from PostgreSQL", errorMsg)


def retrieve_roc_image(connection, metric_id):
    try:
        cursor = connection.cursor()

        retrieve_query = "SELECT roc_image FROM metrics WHERE id = %s;"
        cursor.execute(retrieve_query, (metric_id,))
        image_data = cursor.fetchone()[0]
        print("Roc image retrieved successfully from PostgreSQL")
        return image_data
    except (Exception, psycopg2.Error) as errorMsg:
        print("Error while retrieving image from PostgreSQL", errorMsg)


def get_current_user_id(connection, username):
    try:
        cursor = connection.cursor()

        select_query = "SELECT id FROM users WHERE name = %s;"
        cursor.execute(select_query, (username,))
        user_id = cursor.fetchone()[0]

        return user_id
    except (Exception, psycopg2.Error) as errorMsg:
        print("Error while getting current user ID from PostgreSQL", errorMsg)


def get_id_by_metric_name(connection, metric_name):
    try:
        cursor = connection.cursor()

        select_query = "SELECT id FROM metrics WHERE metric_name = %s;"
        cursor.execute(select_query, (metric_name,))
        result = cursor.fetchone()

        if result:
            metric_id = result[0]
            return metric_id
        else:
            print("Metric not found in the metrics table.")
            return None
    except (Exception, psycopg2.Error) as errorMsg:
        print("Error while retrieving metric_id by metric_name from PostgreSQL", errorMsg)


def check_metric_exists(connection, metric_name):
    cursor = connection.cursor()
    select_query = "SELECT EXISTS(SELECT 1 FROM metrics WHERE metric_name = %s);"
    cursor.execute(select_query, (metric_name,))
    metric_exists = cursor.fetchone()[0]
    return metric_exists