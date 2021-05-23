import requests
import json
import csv
import datetime
import os
import os.path
from os import path
import logging
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd

# CONSTANTS
URLSTRING_VACCINE_BY_DISTRICT = "https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/calendarByDistrict?district_id={district_id}&date={datestr}" 
CSV_FILE_NAME = '/Users/vaibhavb/Desktop/repos/dump/mle_applications/Challenges/cowin_api_apps/vaccine_availability.csv'
MIN_DATE_TO_CHECK = '23-05-2021'
DOSE_TYPE_TO_CHECK = 'available_capacity_dose2'
DOSE_MIN_AGE = 45
VACCINE_TYPE = 'COVAXIN'
DISTRICT_MAP = {
	'Gurgaon' : 188,
	'New_Delhi' : 140,
	"Central Delhi" : 141,
	"North Delhi" : 146,
	"South Delhi" : 149,
	"South West Delhi" : 150,
	"North West Delhi" : 143,
	"East Delhi" : 145,
	"West Delhi" : 142
}

DISTRICT_ID_MAP = {v: k for k, v in DISTRICT_MAP.items()}
headers = {
	'accept': 'application/json', 
	'Accept-Charset': 'UTF-8',
	'Accept-Language': 'hi_IN',
	'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
}

# Set Logging
dir_path = os.path.dirname(os.path.realpath(__file__))
LOG_FILE_NAME = os.path.join(dir_path, '/Users/vaibhavb/Desktop/repos/dump/mle_applications/Challenges/cowin_api_apps/test_log.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def do_logging(message):
    logger.info(message)

def get_routine(get_request_url_string, headers):
	r = requests.get(
		get_request_url_string, 
		headers=headers
	)
	return r

def get_vaccine_metrics_by_district(district_id, datestr):
	response_dict_json = get_routine(
			URLSTRING_VACCINE_BY_DISTRICT.format(
					district_id=district_id,
					datestr=datestr
				), 
			headers
		).json()
	
	total_capacity = 0
	num_vaccine_type = 0
	vaccine_type_freq = {}
	center_name_set = set()

	for center in response_dict_json['centers']:

		sessions = center['sessions']
		center_name = center['name']

		for session in sessions:
			if session['min_age_limit'] >= DOSE_MIN_AGE and session['vaccine'] == VACCINE_TYPE:
				session_capacity = session[DOSE_TYPE_TO_CHECK]
				total_capacity += session_capacity
				if session_capacity > 0:
					center_name_set.add(center_name)
					vaccine_type = session['vaccine']
					if vaccine_type not in vaccine_type_freq:
						vaccine_type_freq[vaccine_type] = 0
					vaccine_type_freq[vaccine_type] += session_capacity
		num_vaccine_type = len(vaccine_type_freq)

	return [
		datetime.datetime.now(),
		datestr,
		district_id,
		DISTRICT_ID_MAP[district_id],
		total_capacity,
		num_vaccine_type,
		json.dumps(vaccine_type_freq),
		str(center_name_set)
	]

# # Data Logging
def write_row_vaccine_by_district(data_row):
	if (path.exists(CSV_FILE_NAME) == False):
		with open(CSV_FILE_NAME, 'w', newline='') as file_object:
		    writer = csv.writer(file_object)
		    writer.writerow(["timestamp", "datestr_vaccine_for", "district_id", "district_name", "total_capacity", "num_vaccine_types", "vaccine_type_freq_json", "center_name_set"])
		    file_object.close()

	with open(CSV_FILE_NAME, 'a', newline='') as file_object:
	    writer = csv.writer(file_object)
	    writer.writerow(data_row)
	    file_object.close()

def mailer(email_html_content, subject):

	# Get mail server secrets from environment
	port = int(os.environ.get("COVID_SCRIPT_VAR_PORT"))  # For SSL
	smtp_server = os.environ.get("COVID_SCRIPT_VAR_SMPT_SERVER")
	sender_email = os.environ.get("COVID_SCRIPT_VAR_SENDER_EMAIL")  # Enter your address
	receiver_email = os.environ.get("COVID_SCRIPT_VAR_RECEIVER_EMAIL")  # Enter receiver address
	password = os.environ.get("COVID_SCRIPT_VAR_PASSWORD")

	message = MIMEMultipart("alternative")
	message["Subject"] = subject
	message["From"] = sender_email
	message["To"] = receiver_email

	# Create the plain-text and HTML version of your message
	message.attach(MIMEText(email_html_content, "html"))

	context = ssl.create_default_context()
	with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
		server.login(sender_email, password)
		server.sendmail(sender_email, receiver_email, message.as_string())


def main():
	do_logging("started")

	# Max of today or user_defined date on when to check
	today_datestr = max(datetime.datetime.today(), datetime.datetime.strptime(MIN_DATE_TO_CHECK, '%d-%m-%Y')).strftime('%d-%m-%Y')
	email_html_content = """\
					<html>
					  <body>
					   	<div>{email_content}</div> 
					  </body>
					</html>
					"""
	district_content_accumulated = ""
	total_slots = 0

	for city in DISTRICT_MAP.values():

		data_row = get_vaccine_metrics_by_district(city, today_datestr)
		write_row_vaccine_by_district(data_row)

		district_slots_available = data_row[4]
		total_slots += district_slots_available

		if district_slots_available > 0:

			# Other cutom alerts such as increment in doses
			# vaccine_db = pd.read_csv(CSV_FILE_NAME)
			# last_vaccine_count = json.loads(vaccine_db[vaccine_db['district_name']=='Gurgaon'].iloc[-2]['vaccine_type_freq_json']).get('COVAXIN') or 0
			# new_vaccine_count = (json.loads(data_row[-2]).get('COVAXIN') or 0)

			if True:
				district_content_accumulated += """\
					   	<b style='color:#ef7b61;'>{city_name}</b> 
					   	<div style='color:#c64022;'>{slots_info}</div> 
					   	<div style='color:#31738a;'>{center_names}</div> 
					   	<br>
					   	<br>
					""".format(city_name=str(data_row[3]), slots_info=data_row[-2], center_names=data_row[-1])
	if total_slots > 0:
		subject = "{vaccine_type} Vaccine Slots Available".format(vaccine_type=str(VACCINE_TYPE))
		mailer(email_html_content.format(email_content=district_content_accumulated), subject)

	do_logging("exiting")
main()

# crontab syntax
# */3 * * * * EXPORT ENV_VAR="xx"; /Users/vaibhavb/miniconda3/bin/python /Users/vaibhavb/Desktop/repos/mle_applications/Challenges/cowin_api_apps/vaccine_fetch.py
