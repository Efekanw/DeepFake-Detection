import psycopg2


def connect(dbname, user, password):
    try:
        print("Connecting to database...")
        connection = psycopg2.connect("dbname =" + dbname + " user = " + user + " password = " + password)
        print("Connection to the database has been established.")
        return connection
    except(Exception, psycopg2.Error) as errorMsg:
        print("A database-related error occured: ", errorMsg)
        return None


def disconnect(connection):
    if connection:
        connection.close()
        print("\nPostgreSQL connection is closed\n")