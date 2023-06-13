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