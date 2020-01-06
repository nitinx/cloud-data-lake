import os
import datetime
import configparser
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf, col, to_timestamp, to_date, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    '''Function to create a spark session'''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''Function to process song data and generate parquet files for each dimension'''
    # get filepath to song data file
    data_song = input_data + "song_data/*/*/*/*.json"
    
    # read song data file
    df_song = spark.read.json(data_song)

    # extract columns to create songs table
    songs_table = df_song.select(
        'song_id', 'title', 'artist_id', 'year', 'duration').dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year","artist_id").parquet(output_data + "songs.parquet")

    # extract columns to create artists table
    artists_table = df_song.select(
        'artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude').dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + "artists.parquet")


def process_log_data(spark, input_data, output_data):
    '''Function to process log data and generate parquet files for the fact and dimensions'''
    # get filepath to log data file
    data_log = input_data + "log-data/*/*/*.json"

    # read log data file
    df_log = spark.read.json(data_log)
    
    # filter by actions for song plays
    df_log = df_log.filter(df_log.page == 'NextSong')
    df_log = df_log.withColumn("songplay_id", monotonically_increasing_id())

    # extract columns for users table    
    users_table = df_log.select(
        'userId', 'firstName', 'lastName', 'gender', 'level').dropDuplicates()
    
    # write users table to parquet files
    users_table.write.parquet(output_data + "users.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime('%H:%M:%S'))
    df_log = df_log.withColumn("timestamp", get_timestamp(df_log.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d'))
    df_log = df_log.withColumn("datetime", get_datetime(df_log.ts))
    
    get_weekday = udf(lambda day_number: 1 if day_number < 6 else 0)
    
    # extract columns to create time table
    time_table = df_log.select(
        'timestamp',
        hour(to_timestamp('timestamp', 'HH:mm:ss')).alias('hour'),
        dayofmonth(to_date('datetime', 'yyyy-MM-dd')).alias('day'),
        weekofyear(to_date('datetime', 'yyyy-MM-dd')).alias('week'),
        month(to_date('datetime', 'yyyy-MM-dd')).alias('month'),
        year(to_date('datetime', 'yyyy-MM-dd')).alias('year'),
        date_format('datetime', 'u').cast(IntegerType()).alias('day_number')
    ).dropDuplicates()
    
    time_table = time_table.withColumn("weekday", get_weekday(time_table.day_number))
    time_table = time_table.withColumnRenamed("timestamp", "start_time")
    time_table = time_table.drop('day_number')
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year","month").parquet(output_data + "time.parquet")

    # read in song data to use for songplays table
    songs_table = spark.read.parquet(output_data + "songs.parquet")
    
    df_log = df_log.alias('df_log')
    songs_table = songs_table.alias('songs_table')
    time_table = time_table.alias('time_table')    

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df_log.join(songs_table, col('df_log.song') == col('songs_table.title'), 'left') \
    .join(time_table, col('df_log.timestamp') == col('time_table.start_time'), 'left') \
    .select('df_log.songplay_id', 
            'df_log.timestamp', 
            'df_log.userId', 
            'df_log.level', 
            'songs_table.song_id', 
            'songs_table.artist_id', 
            'df_log.sessionId', 
            'df_log.location', 
            'df_log.userAgent', 
            'time_table.year', 
            'time_table.month')

    songplays_table = (songplays_table.withColumnRenamed("timestamp", "start_time") \
                       .withColumnRenamed("userId", "user_id") \
                       .withColumnRenamed("sessionId", "session_id") \
                       .withColumnRenamed("userAgent", "user_agent"))

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year","month").parquet(output_data + "songplays.parquet")


def main():
    spark = create_spark_session()
    #input_data = "s3a://udacity-dend/"
    input_data = "data/"
    
    output_data = "s3a://sparkify-dl/output/"
    #output_data = "etl/output/"

    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)

    
if __name__ == "__main__":
    main()
