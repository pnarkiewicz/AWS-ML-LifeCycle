import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import os

s3 = boto3.client("s3")


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_file(file_name: str, bucket: str):
    """Download a file and save it to temp folder

    :param file_name: File to download
    :param bucket: Bucket to download from
    :return: True if file has been downloaded, else False
    """

    try:
        s3.download_file(bucket, file_name, f"temp/{file_name.split('/')[-1]}")
    except ClientError as e:
        logging.error(e)
        return False


def download_return_csv_dataframe(file_name: str, bucket: str):
    """Download csv file and return pandas Dataframe

    :param file_name: File to download
    :param bucket: Bucket to download from
    :return: csv file in a dataframe format
    """

    download_file(file_name, bucket)
    return pd.read_csv(f"temp/{file_name.split('/')[-1]}")
